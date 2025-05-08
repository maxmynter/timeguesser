# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///


import json
import os
import re

# Input/output files
INPUT_FILE = "historical_events.json"  # Your unified event dataset
OUTPUT_FILE = "enhanced_events.json"  # Enhanced dataset with titles


def extract_title(text, event_type):
    """Extract a concise title from the event text"""
    # For births, the title is typically the person's name
    if event_type == "birth":
        # Pattern to match: "Name, description"
        match = re.match(r"([^,]+)(?:,|$)", text)
        if match:
            return f"{match.group(1)} born"
        return "Birth: " + text[:30] + "..."

    # For deaths, similar approach
    elif event_type == "death":
        match = re.match(r"([^,]+)(?:,|$)", text)
        if match:
            return f"{match.group(1)} dies"
        return "Death: " + text[:30] + "..."

    # For general events, it's trickier
    else:
        # If it's short, use as is
        if len(text) < 60:
            return text

        # Try to extract the main event
        # Look for sentences or clauses
        sentences = re.split(r"[.;:]", text)
        if sentences:
            return sentences[0]

        # Fallback: just truncate
        return text[:60] + "..."


def create_embedding_text(event):
    """Create text optimized for embedding"""
    # Prefix the type for better semantic retrieval
    prefix = {"birth": "Born: ", "death": "Died: ", "event": "Event: "}.get(
        event["type"], ""
    )

    # Include year for temporal context
    year_text = f"In {event['year_original']}, " if event["year_original"] else ""

    # Combine for the embedding text
    return f"{prefix}{year_text}{event['text']}"


def enhance_events():
    """Enhance events with titles and embedding text"""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Load the events
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data["events"]
    enhanced_events = []

    print(f"Enhancing {len(events)} events...")

    for event in events:
        # Extract a title
        title = extract_title(event["text"], event["type"])

        # Create embedding-optimized text
        embedding_text = create_embedding_text(event)

        # Enhanced event
        enhanced_event = {
            **event,  # Keep all original fields
            "title": title,
            "embedding_text": embedding_text,
        }

        enhanced_events.append(enhanced_event)

    # Save the enhanced events
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "events": enhanced_events,
                "metadata": {**data.get("metadata", {}), "enhanced": True},
            },
            f,
            indent=2,
        )

    print(f"Enhanced {len(enhanced_events)} events and saved to {OUTPUT_FILE}")

    # Print a few examples
    print("\nExample titles:")
    for i in range(min(5, len(enhanced_events))):
        print(f"Original: {enhanced_events[i]['text']}")
        print(f"Title: {enhanced_events[i]['title']}")
        print(f"Embedding: {enhanced_events[i]['embedding_text']}")
        print("---")


if __name__ == "__main__":
    enhance_events()
