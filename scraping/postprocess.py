# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#    "typer",
#    "pydantic>1.10.2",
#    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#    "spacy>=3.7",
#    ]
# ///

import json
import spacy
import typer
from pathlib import Path
from typing import Dict, List, Any, Optional

# Initialize CLI app
app = typer.Typer(help="Process historical events data")


def func(sd: int):
    pass


def load_json(file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load JSON data from file

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the loaded JSON data
    """
    try:
        typer.echo(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate expected structure
        if not isinstance(data, dict) or "events" not in data:
            typer.echo(
                "Warning: Input file doesn't have an 'events' key at the top level."
            )

        # Count events for reporting
        event_count = len(data.get("events", []))
        typer.echo(f"Loaded {event_count} events")

        return data

    except json.JSONDecodeError:
        typer.echo(f"Error: {file_path} is not a valid JSON file.")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error loading file: {e}")
        raise typer.Exit(code=1)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save JSON data to file

    Args:
        data: Dictionary containing the processed data
        file_path: Path where the JSON file will be saved
    """
    try:
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Saving data to {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        typer.echo(f"Successfully saved data with {len(data.get('events', []))} events")

    except Exception as e:
        typer.echo(f"Error saving file: {e}")
        raise typer.Exit(code=1)


def process_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process the events data - customize this function for your specific needs

    Args:
        events: List of event dictionaries

    Returns:
        Processed list of event dictionaries
    """
    nlp = spacy.load("en_core_web_sm")
    for i, event in enumerate(events):
        if i % 1000 == 0:
            print(i)
        title = nlp(event["title"])
        text = nlp(event["text"])
        processed_title = " ".join([tok.lemma_ for tok in title if not tok.is_stop])
        processed_text = " ".join([tok.lemma_ for tok in text if not tok.is_stop])
        entities = list(
            set([ent.lemma_ for ent in title.ents] + [ent.lemma_ for ent in text.ents])
        )
        event["lemmatized_title"] = processed_title
        event["lemmatized_text"] = processed_text
        event["entities"] = entities
    return events


@app.command()
def process(
    source: Path = typer.Option(
        ..., "--source", "-s", help="Source JSON file with historical events"
    ),
    target: Path = typer.Option(
        ..., "--target", "-t", help="Target JSON file to save processed events"
    ),
):
    """
    Process historical events JSON file
    """
    # Verify source file exists
    if not source.exists():
        typer.echo(f"Error: Source file {source} not found.")
        raise typer.Exit(code=1)

    # Load data
    data = load_json(source)

    # Process events
    if "events" in data:
        data["events"] = process_events(data["events"])
    else:
        typer.echo("Warning: No 'events' key found in the input data.")

    # Show success message
    typer.echo("Processing completed successfully!")
    save_json(data, target)


if __name__ == "__main__":
    app()
