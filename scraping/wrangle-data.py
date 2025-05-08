# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

import json
import os
import re
import glob
from datetime import datetime

# Input/output directories
INPUT_DIR = "data"  # Directory containing the original JSON files
OUTPUT_FILE = "historical_events.json"  # Single output file for all events


def parse_year(year_str):
    """Convert year string to integer, handling BC/BCE dates"""
    year_str = year_str.strip()
    if "BC" in year_str or "BCE" in year_str:
        # Remove BC/BCE and convert to negative number
        numeric_part = re.search(r"\d+", year_str)
        if numeric_part:
            return -int(numeric_part.group(0))
        return None
    else:
        # Handle CE/AD years
        numeric_part = re.search(r"\d+", year_str)
        if numeric_part:
            return int(numeric_part.group(0))
        return None


def extract_month_day(filename):
    """Extract month and day from filename (e.g., '01-01.json' -> (1, 1))"""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]  # Remove extension
    month_str, day_str = name.split("-")
    return int(month_str), int(day_str)


def process_json_files():
    """Process all JSON files and create a unified data structure"""
    all_events = []

    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    for file_path in json_files:
        print(f"Processing {file_path}...")

        try:
            # Extract month and day from filename
            month, day = extract_month_day(file_path)

            # Load the JSON data
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Process each category (Events, Births, Deaths)
            for category in ["Events", "Births", "Deaths"]:
                if category in data["data"]:
                    for entry in data["data"][category]:
                        # Parse the year
                        year_int = parse_year(entry["year"])

                        # Create standardized entry
                        event = {
                            "type": category.lower()[
                                :-1
                            ],  # 'event', 'birth', or 'death'
                            "year": year_int,
                            "year_original": entry["year"],  # Keep original string
                            "month": month,
                            "day": day,
                            "text": entry["text"],
                            "links": entry["links"],
                            # Optional - add these if you need them
                            # 'html': entry['html'],
                            # 'no_year_html': entry['no_year_html'],
                        }

                        # Add to the collection
                        all_events.append(event)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Processed {len(all_events)} events total")

    # Sort by year, month, day
    all_events.sort(key=lambda x: (x["year"] or -9999, x["month"], x["day"]))

    # Save to a single JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "events": all_events,
                "metadata": {
                    "total_count": len(all_events),
                    "generated_at": datetime.now().isoformat(),
                    "source": "Wikipedia",
                },
            },
            f,
            indent=2,
        )

    print(f"Saved all events to {OUTPUT_FILE}")

    # Generate some statistics
    event_types = {}
    for event in all_events:
        event_type = event["type"]
        event_types[event_type] = event_types.get(event_type, 0) + 1

    print("\nEvent type statistics:")
    for event_type, count in event_types.items():
        print(f"- {event_type}: {count}")


if __name__ == "__main__":
    process_json_files()
