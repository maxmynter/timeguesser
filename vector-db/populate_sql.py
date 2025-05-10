#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "supabase",
#   "python-dotenv"
# ]
# ///


import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import uuid
from dataclasses import dataclass
import argparse
from functools import partial
import logging
from dotenv import load_dotenv

# Import supabase client
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EventData:
    """Represents core event data"""

    type: str
    year: int
    year_original: Optional[str]
    month: Optional[int]
    day: Optional[int]
    text: str
    title: str
    embedding_text: Optional[str]
    lemmatized_title: Optional[str]
    lemmatized_text: Optional[str]


@dataclass
class Link:
    """Represents a link associated with an event"""

    title: str
    url: str


@dataclass
class Entity:
    """Represents an entity associated with an event"""

    name: str


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Import events from JSON into Supabase PostgreSQL"
    )
    parser.add_argument("json_file", help="Path to the JSON file containing events")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of events to process in a batch",
    )
    return parser.parse_args()


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load events from a JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "events" not in data:
            logger.error(f"Invalid JSON format: missing 'events' key")
            sys.exit(1)

        return data["events"]
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading JSON file: {e}")
        sys.exit(1)


def parse_event(
    event_json: Dict[str, Any],
) -> Tuple[EventData, List[Link], List[Entity]]:
    """Parse an event JSON object into structured data"""
    # Extract core event data
    event = EventData(
        type=event_json.get("type", "event"),
        year=event_json.get("year"),
        year_original=event_json.get("year_original"),
        month=event_json.get("month"),
        day=event_json.get("day"),
        text=event_json.get("text", ""),
        title=event_json.get("title", ""),
        embedding_text=event_json.get("embedding_text"),
        lemmatized_title=event_json.get("lemmatized_title"),
        lemmatized_text=event_json.get("lemmatized_text"),
    )

    # Extract links
    links = []
    for link_data in event_json.get("links", []):
        links.append(
            Link(title=link_data.get("title", ""), url=link_data.get("link", ""))
        )

    # Extract entities
    entities = []
    for entity_name in event_json.get("entities", []):
        entities.append(Entity(name=entity_name))

    return event, links, entities


def initialize_supabase() -> Client:
    """Initialize Supabase client"""
    # Load environment variables
    load_dotenv()

    # Get Supabase URL and key from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("Missing SUPABASE_URL or SUPABASE_KEY environment variables")
        sys.exit(1)

    return create_client(supabase_url, supabase_key)


def insert_event_batch(
    supabase: Client, event_batch: List[Tuple[EventData, List[Link], List[Entity]]]
) -> None:
    """Insert a batch of events into the database"""
    try:
        # Group data by table
        events_data = []
        entities_data = []
        links_data = []

        for event, event_links, event_entities in event_batch:
            # Generate a UUID for the event
            event_id = str(uuid.uuid4())

            # Prepare event data
            event_row = {
                "id": event_id,
                "type": event.type,
                "year": event.year,
                "year_original": event.year_original,
                "month": event.month,
                "day": event.day,
                "text": event.text,
                "title": event.title,
                "embedding_text": event.embedding_text,
                "lemmatized_title": event.lemmatized_title,
                "lemmatized_text": event.lemmatized_text,
            }
            events_data.append(event_row)

            # Prepare entity data
            for entity in event_entities:
                entities_data.append(
                    {
                        "id": str(uuid.uuid4()),
                        "event_id": event_id,
                        "entity_name": entity.name,
                    }
                )

            # Prepare link data
            for link in event_links:
                links_data.append(
                    {
                        "id": str(uuid.uuid4()),
                        "event_id": event_id,
                        "title": link.title,
                        "url": link.url,
                    }
                )

        # Insert events
        if events_data:
            supabase.table("events").insert(events_data).execute()

        # Insert entities
        if entities_data:
            supabase.table("entities").insert(entities_data).execute()

        # Insert links
        if links_data:
            supabase.table("links").insert(links_data).execute()

    except Exception as e:
        logger.error(f"Error inserting batch: {e}")
        raise


def process_events(
    supabase: Client, events: List[Dict[str, Any]], batch_size: int
) -> None:
    """Process all events from the JSON file and insert them into the database"""
    total_events = len(events)
    logger.info(f"Processing {total_events} events")

    # Parse events
    parsed_events = []
    for event_json in events:
        try:
            parsed_event = parse_event(event_json)
            parsed_events.append(parsed_event)
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")

    # Process in batches
    for i in range(0, len(parsed_events), batch_size):
        batch = parsed_events[i : i + batch_size]
        try:
            insert_event_batch(supabase, batch)
            logger.info(
                f"Inserted batch {i // batch_size + 1}/{(len(parsed_events) + batch_size - 1) // batch_size}"
            )
        except Exception as e:
            logger.error(f"Failed to insert batch starting at index {i}: {e}")


def main() -> None:
    """Main entry point"""
    args = parse_arguments()

    # Initialize Supabase client
    supabase = initialize_supabase()

    # Load and process events
    events = load_json_data(args.json_file)

    # Process events in batches
    process_events(supabase, events, args.batch_size)

    logger.info("Import completed successfully")


if __name__ == "__main__":
    main()
