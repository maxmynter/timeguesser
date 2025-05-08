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

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "historical_events"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384  # Embedding dimension
INPUT_FILE = "enhanced_events.json"  # Default input file

# Initialize FastAPI app
app = FastAPI(
    title="TimeQuest API",
    description="API for searching historical events with semantic search",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
logger.info(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Initialize Qdrant client
# Connect to the running Qdrant server instead of memory
logger.info("Connecting to Qdrant server")
client = QdrantClient(host="localhost", port=6333)


# Define response models
class Link(BaseModel):
    title: str
    link: str


class HistoricalEvent(BaseModel):
    id: int
    title: str
    text: str
    year: Optional[int] = None
    year_original: str
    month: int
    day: int
    type: str
    links: List[Link]
    score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[HistoricalEvent]
    total: int
    query: str


class ErrorResponse(BaseModel):
    error: str
    query: Optional[str] = None
    results: List = []
    total: int = 0


# Utility to load events from JSON if starting fresh
def load_events_from_json(file_path: str = INPUT_FILE) -> bool:
    """Load events from JSON file and populate the database"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    try:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("events", [])
        if not events:
            logger.error("No events found in the JSON file")
            return False

        logger.info(f"Found {len(events)} events in the JSON file")

        # Create collection if needed
        try:
            client.get_collection(COLLECTION_NAME)
            logger.info(f"Collection {COLLECTION_NAME} already exists")
        except Exception as e:
            logger.info(f"Creating collection {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        # Generate embeddings and upload in batches
        batch_size = 100
        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]

            # Extract text for embedding
            batch_texts = []
            for event in batch:
                # Use embedding_text if available, otherwise use text
                text = event.get("embedding_text")
                if text is None:
                    text = event.get("text", "")
                batch_texts.append(text)

            # Generate embeddings
            logger.info(
                f"Generating embeddings for batch {i // batch_size + 1}/{(len(events) - 1) // batch_size + 1}"
            )
            batch_embeddings = model.encode(batch_texts)

            # Prepare payloads
            payloads = []
            for event in batch:
                payloads.append(
                    {
                        # Remove the 'id' field if it exists to avoid conflicts
                        "title": event.get("title", ""),
                        "text": event.get("text", ""),
                        "year": event.get("year"),
                        "year_original": event.get("year_original", ""),
                        "month": event.get("month"),
                        "day": event.get("day"),
                        "type": event.get("type", ""),
                        "links": event.get("links", []),
                    }
                )

            # Upsert to database
            logger.info(
                f"Uploading batch {i // batch_size + 1}/{(len(events) - 1) // batch_size + 1} to Qdrant"
            )
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=models.Batch(
                    ids=list(range(i, i + len(batch))),
                    vectors=batch_embeddings,
                    payloads=payloads,
                ),
            )

        logger.info(f"Successfully loaded {len(events)} events to Qdrant")
        return True

    except Exception as e:
        logger.error(f"Error loading events: {str(e)}")
        return False


# Check if collection exists or load data
@app.on_event("startup")
async def startup_event():
    try:
        # Check if collection exists
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            logger.info(
                f"Collection {COLLECTION_NAME} already exists with {collection_info.vectors_count} vectors"
            )

            # If collection exists but is empty, try to load data
            if collection_info.vectors_count == 0:
                logger.info("Collection exists but is empty, trying to load data")
                if load_events_from_json():
                    logger.info("Data loaded successfully")
                else:
                    logger.warning("No data file found or error loading data")

        except Exception as e:
            logger.info(f"Collection error: {str(e)}")
            logger.info(f"Creating collection {COLLECTION_NAME} and loading data")

            # Create collection
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

            # Load data
            if load_events_from_json():
                logger.info("Data loaded successfully")
            else:
                logger.warning(
                    "No data file found or error loading data. Please run the setup scripts first."
                )

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")


# Define API routes
@app.get("/", response_model=Dict[str, str])
def read_root():
    """Root endpoint - provides basic API information"""
    return {"message": "Welcome to TimeQuest API"}


@app.get("/search", response_model=Union[SearchResponse, ErrorResponse])
def search_events(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    type_filter: Optional[str] = Query(
        None, description="Filter by event type (event, birth, death)"
    ),
    year_min: Optional[int] = Query(None, description="Minimum year"),
    year_max: Optional[int] = Query(None, description="Maximum year"),
):
    """
    Search for historical events using semantic search.

    - Accepts natural language queries
    - Can filter by event type and year range
    - Returns semantically relevant results
    """
    try:
        # Generate embedding for the query
        logger.info(f"Searching for: {q}")
        query_vector = model.encode(q).tolist()

        # Prepare filters if needed
        filter_query = None
        if type_filter or year_min is not None or year_max is not None:
            must_conditions = []

            if type_filter:
                must_conditions.append(
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value=type_filter),
                    )
                )

            if year_min is not None or year_max is not None:
                range_condition = models.Range(
                    gte=year_min if year_min is not None else None,
                    lte=year_max if year_max is not None else None,
                )
                must_conditions.append(
                    models.FieldCondition(
                        key="year",
                        range=range_condition,
                    )
                )

            filter_query = models.Filter(must=must_conditions)

        # Search
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_query,
        )

        # Convert to response format
        results = []
        for hit in search_result:
            try:
                # Make a copy of the payload and handle potential ID conflict
                payload_copy = hit.payload.copy()
                if "id" in payload_copy:
                    del payload_copy["id"]  # Remove id to avoid conflict

                # Create HistoricalEvent object
                event = HistoricalEvent(id=hit.id, score=hit.score, **payload_copy)
                results.append(event)
            except Exception as e:
                logger.error(f"Error processing search result: {str(e)}")
                # Continue with next result

        return SearchResponse(results=results, total=len(results), query=q)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return ErrorResponse(error=str(e), query=q)


@app.get("/event/{event_id}", response_model=Union[HistoricalEvent, Dict[str, str]])
def get_event(event_id: int):
    """Get a specific historical event by ID"""
    try:
        # Retrieve the event from Qdrant
        result = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[event_id],
        )

        if not result:
            return JSONResponse(
                status_code=404,
                content={"error": f"Event with id {event_id} not found"},
            )

        # Handle potential ID conflict
        payload_copy = result[0].payload.copy()
        if "id" in payload_copy:
            del payload_copy["id"]

        # Return the event
        return HistoricalEvent(id=event_id, **payload_copy)

    except Exception as e:
        logger.error(f"Error retrieving event {event_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict[str, Any])
def get_stats():
    """Get statistics about the vector database"""
    try:
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)

        # Return stats
        return {
            "total_events": collection_info.vectors_count,
            "vector_dimension": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance),
            "status": "ready",
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"error": str(e), "status": "not_ready"}


# Run the server
if __name__ == "__main__":
    uvicorn.run("scrappy-server:app", host="0.0.0.0", port=8000, reload=True)
