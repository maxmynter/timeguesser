# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "requests",
#   "argparse"
# ]
# ///

import requests
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Query the TimeQuest API")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    parser.add_argument(
        "--type", "-t", choices=["event", "birth", "death"], help="Filter by event type"
    )
    parser.add_argument("--year-min", type=int, help="Minimum year")
    parser.add_argument("--year-max", type=int, help="Maximum year")
    parser.add_argument(
        "--server", "-s", default="http://localhost:8000", help="API server URL"
    )
    parser.add_argument(
        "--pretty", "-p", action="store_true", help="Pretty print the results"
    )
    return parser.parse_args()


def query_api(args):
    # Construct the API URL with query parameters
    url = f"{args.server}/search"
    params = {"q": args.query, "limit": args.limit}

    # Add optional filters if provided
    if args.type:
        params["type_filter"] = args.type
    if args.year_min is not None:
        params["year_min"] = args.year_min
    if args.year_max is not None:
        params["year_max"] = args.year_max

    # Make the request
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying API: {e}")
        return None


def display_results(data, pretty=False):
    if not data:
        print("No results returned")
        return

    if pretty:
        # Pretty print the results
        print(f"\nSearch query: '{data['query']}'")
        print(f"Found {data['total']} results\n")

        for i, result in enumerate(data["results"]):
            print(f"Result #{i + 1} (Score: {result['score']:.4f})")
            print(f"Title: {result['title']}")

            # Format date
            date_str = (
                f"{result['month']}/{result['day']}/{result['year']}"
                if result["year"]
                else "Unknown date"
            )
            print(f"Date: {date_str}")

            print(f"Type: {result['type']}")
            print(f"Text: {result['text']}")

            if result["links"]:
                print("Links:")
                for link in result["links"]:
                    print(f"  - {link['title']}: {link['link']}")

            print("\n" + "-" * 50 + "\n")
    else:
        # Just print the JSON
        print(json.dumps(data, indent=2))


def main():
    args = parse_args()
    results = query_api(args)
    display_results(results, args.pretty)


if __name__ == "__main__":
    main()
