# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "numpy",
#     "pydantic",
#     "qdrant-client",
#     "sentence-transformers",
#     "tqdm",
#     "uvicorn",
#     "argparse",
# ]
# ///

import json
import os
import argparse
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from tqdm import tqdm

# Default configuration
DEFAULT_INPUT_FILE = "enhanced_events.json"
COLLECTION_NAME = "historical_events"
VECTOR_SIZE = 384  # Dimension of the embeddings
BATCH_SIZE = 100  # Process this many events at once

# Use a lightweight model for embeddings
# This model is relatively small but performs well for semantic search
MODEL_NAME = "all-MiniLM-L6-v2"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create vector embeddings for historical events"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to enhanced events JSON file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default=COLLECTION_NAME,
        help=f"Name of the Qdrant collection (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=MODEL_NAME,
        help=f"Name of the sentence transformer model (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for processing (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use in-memory Qdrant database (default: true)",
    )
    parser.add_argument(
        "--server", "-s", help="Qdrant server address (format: host:port)"
    )
    parser.add_argument(
        "--test-query",
        "-q",
        action="append",
        default=["Moon landing", "French Revolution", "Albert Einstein birth"],
        help="Test queries to run after embedding (can specify multiple times)",
    )

    return parser.parse_args()


def load_events(input_file):
    """Load the enhanced events"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["events"]


def setup_qdrant(args):
    """Setup the Qdrant vector database"""
    if args.server:
        # Connect to remote Qdrant server
        host, port = args.server.split(":")
        client = QdrantClient(host=host, port=int(port))
        print(f"Connected to Qdrant server at {args.server}")
    else:
        # Use in-memory Qdrant
        client = QdrantClient(location=":memory:")
        print("Using in-memory Qdrant database")

    # Create the collection if it doesn't exist
    try:
        client.get_collection(args.collection)
        print(f"Collection {args.collection} already exists")
    except Exception:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created collection {args.collection}")

    return client


def embed_events(events, model, batch_size):
    """Generate embeddings for all events"""
    embeddings = []
    texts = []

    print(f"Generating embeddings for {len(events)} events...")

    # Process events in batches
    for i in tqdm(range(0, len(events), batch_size)):
        batch = events[i : i + batch_size]
        batch_texts = [
            event.get("embedding_text", event.get("text", "")) for event in batch
        ]

        # Generate embeddings
        batch_embeddings = model.encode(batch_texts)

        embeddings.extend(batch_embeddings)
        texts.extend(batch_texts)

    return embeddings, texts


def populate_database(client, events, embeddings, collection_name, batch_size):
    """Populate the Qdrant database with event embeddings"""
    print("Populating database...")

    # Prepare payloads (metadata)
    payloads = []
    for event in events:
        payload = {
            "id": event.get("id", None),
            "title": event.get("title", ""),
            "text": event.get("text", ""),
            "year": event.get("year"),
            "year_original": event.get("year_original", ""),
            "month": event.get("month"),
            "day": event.get("day"),
            "type": event.get("type", ""),
            "links": event.get("links", []),
        }
        payloads.append(payload)

    # Upload in batches
    for i in tqdm(range(0, len(events), batch_size)):
        batch_payloads = payloads[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        batch_ids = list(range(i, i + len(batch_payloads)))

        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=batch_ids,
                vectors=batch_embeddings,
                payloads=batch_payloads,
            ),
        )

    print(f"Uploaded {len(events)} events to Qdrant collection '{collection_name}'")


def test_search(client, model, collection_name, query):
    """Test searching the database"""
    print(f"\nTesting search with query: '{query}'")

    # Generate embedding for the query
    query_embedding = model.encode(query)

    # Search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5,
    )

    print("\nTop 5 results:")
    for i, result in enumerate(search_result):
        print(f"{i + 1}. Score: {result.score:.4f}")
        print(f"   Title: {result.payload.get('title', 'No title')}")

        # Format date if available
        year = result.payload.get("year")
        month = result.payload.get("month")
        day = result.payload.get("day")
        if all(x is not None for x in [year, month, day]):
            print(f"   Date: {month}/{day}/{year}")

        print(f"   Type: {result.payload.get('type', 'Unknown')}")
        text = result.payload.get("text", "No text")
        print(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
        print()


def main():
    """Main function to set up the vector database"""
    # Parse command line arguments
    args = parse_arguments()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Please specify a valid input file.")
        return

    try:
        # Load events
        events = load_events(args.input)
        print(f"Loaded {len(events)} events from {args.input}")

        # Setup Qdrant
        client = setup_qdrant(args)

        # Load embedding model
        print(f"Loading embedding model {args.model}...")
        model = SentenceTransformer(args.model)

        # Generate embeddings
        embeddings, texts = embed_events(events, model, args.batch_size)

        # Populate database
        populate_database(client, events, embeddings, args.collection, args.batch_size)

        # Test queries
        for query in args.test_query:
            test_search(client, model, args.collection, query)

        print("\nVector database setup complete!")

    except Exception as e:
        print(f"Error setting up vector database: {e}")


if __name__ == "__main__":
    main()
