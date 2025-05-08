# /// script
# requires-python = ">=3.13"
# dependencies = [
#    "requests"
#  ]
# ///


import json
import time
from datetime import datetime, timedelta
import os
import re
from urllib.parse import urlparse, quote
import requests

# Input/output files
INPUT_FILE = "historical_events.json"  # Your unified event dataset
OUTPUT_FILE = "popularity_enhanced_events.json"  # Enhanced dataset with titles


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


def get_pageviews_for_article(title, retries=3, backoff_factor=2):
    """
    Get pageviews for a Wikipedia article for the last year

    Args:
        title: The title of the Wikipedia article (from the URL)
        retries: Number of retries on failure
        backoff_factor: Multiplicative factor for backoff between retries

    Returns:
        Total pageviews for the last year or 0 if failed
    """
    # Calculate date range for the last year (from 1 year ago to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Format dates for the API
    start_str = start_date.strftime("%Y%m%d00")
    end_str = end_date.strftime("%Y%m%d00")

    # Encode the title for the URL
    encoded_title = quote(title.replace(" ", "_"))

    # Construct the API URL
    api_url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{encoded_title}/monthly/{start_str}/{end_str}"

    current_retry = 0
    while current_retry <= retries:
        try:
            response = requests.get(
                api_url, headers={"User-Agent": "Historical Events Analyzer/1.0"}
            )

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                # Sum up the pageviews for all months
                total_views = sum(
                    item.get("views", 0) for item in data.get("items", [])
                )
                return total_views

            # If the article doesn't exist or has no views
            elif response.status_code == 404:
                print(f"No pageview data found for: {title}")
                return 0

            # If we hit the rate limit or other server error
            elif response.status_code in (429, 500, 503):
                wait_time = backoff_factor**current_retry
                print(
                    f"Rate limited or server error. Waiting {wait_time} seconds before retry. Status: {response.status_code}"
                )
                time.sleep(wait_time)
                current_retry += 1

            # Other errors
            else:
                print(
                    f"Error fetching pageviews for {title}. Status code: {response.status_code}"
                )
                return 0

        except Exception as e:
            print(f"Exception when fetching pageviews for {title}: {str(e)}")
            wait_time = backoff_factor**current_retry
            time.sleep(wait_time)
            current_retry += 1

    # If we've exhausted all retries
    print(f"Failed to fetch pageviews for {title} after {retries} retries")
    return 0


def extract_title_from_wikipedia_url(url):
    """Extract the page title from a Wikipedia URL"""
    parsed_url = urlparse(url)
    if "wikipedia.org" not in parsed_url.netloc:
        return None

    # The title is the last part of the path
    path_parts = parsed_url.path.split("/")
    if len(path_parts) < 3:
        return None

    # The title is URL encoded, so we need to decode it
    title = path_parts[-1]
    return title


def enhance_events():
    """Enhance events with titles, embedding text, and pageviews"""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Load the events
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data["events"]
    enhanced_events = []

    print(f"Enhancing {len(events)} events with pageviews...")

    # Track progress
    total_events = len(events)
    processed = 0

    for event in events:
        # Extract a title
        title = extract_title(event["text"], event["type"])

        # Create embedding-optimized text
        embedding_text = create_embedding_text(event)

        # Process links to get pageviews
        links_with_pageviews = []
        total_pageviews = 0

        for link in event.get("links", []):
            link_url = link.get("link")
            wiki_title = extract_title_from_wikipedia_url(link_url)

            if wiki_title:
                # Add a small delay between requests to respect rate limits
                # time.sleep(0.001)

                pageviews = get_pageviews_for_article(wiki_title)

                # Add pageviews to the link info
                link_with_pageviews = {**link, "pageviews": pageviews}
                links_with_pageviews.append(link_with_pageviews)
                total_pageviews += pageviews
            else:
                # Keep the original link if it's not a Wikipedia URL
                links_with_pageviews.append(link)

        # Enhanced event
        enhanced_event = {
            **event,  # Keep all original fields
            "title": title,
            "embedding_text": embedding_text,
            "links": links_with_pageviews,
            "total_pageviews": total_pageviews,
        }

        enhanced_events.append(enhanced_event)

        # Update progress
        processed += 1
        if processed % 10 == 0:
            print(
                f"Processed {processed}/{total_events} events ({(processed / total_events) * 100:.1f}%)"
            )

        # Periodically save progress to avoid losing work on crashes
        if processed % 100 == 0:
            # Save the enhanced events
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "events": enhanced_events,
                        "metadata": {
                            **data.get("metadata", {}),
                            "enhanced": True,
                            "pageviews_added": True,
                            "processed_count": processed,
                            "total_count": total_events,
                            "last_update": datetime.now().isoformat(),
                        },
                    },
                    f,
                    indent=2,
                )
            print(f"Saved progress after processing {processed} events")

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "events": enhanced_events,
                "metadata": {
                    **data.get("metadata", {}),
                    "enhanced": True,
                    "pageviews_added": True,
                    "completed": True,
                    "processed_date": datetime.now().isoformat(),
                },
            },
            f,
            indent=2,
        )

    print(f"Enhanced {len(enhanced_events)} events and saved to {OUTPUT_FILE}")

    # Print a few examples
    print("\nExample events with pageviews:")
    for i in range(min(5, len(enhanced_events))):
        print(f"Title: {enhanced_events[i]['title']}")
        print(f"Total pageviews: {enhanced_events[i]['total_pageviews']}")
        print("Links pageviews:")
        for link in enhanced_events[i]["links"]:
            pageviews = link.get("pageviews", "N/A")
            print(f"  - {link['title']}: {pageviews}")
        print("---")


if __name__ == "__main__":
    enhance_events()
