# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "beautifulsoup4",
#     "requests",
# ]
# ///

import json
import os
import re
from datetime import date, timedelta
import requests
from bs4 import BeautifulSoup

# Constants
DIVIDERS = ["–", "-", "–"]
SECTIONS = ["Events", "Births", "Deaths"]

DEST = os.environ.get("DEST", "data")
CACHE_HTML = os.environ.get("CACHE_HTML", "0") == "1"

EXPECTED_BASE = ["date", "url", "data"]
EXPECTED_DATA = ["Events", "Births", "Deaths"]

MIN_YEAR = 40  # Events should include some from the last 40 years
MIN_EVENTS = 20  # Should have at least 20 events

# Create data directory if it doesn't exist
if not os.path.isdir(DEST):
    os.makedirs(DEST)


def closest_header(element):
    """Find the closest header before the given element"""
    prev = element.find_previous(["h2"])
    while prev:
        if prev.name == "h2":
            return prev
        prev = prev.find_previous(["h2"])
    return None


def process_list(ul, url_base="https://wikipedia.org"):
    """Process a list of historical items"""
    data = []

    for item in ul.find_all("li", recursive=False):
        # Get the flat text of the entry
        text = item.get_text().strip()
        if not text:
            continue

        # Parse the year and content
        year_pattern = re.compile(
            r"^(.*?)(?:\s+[" + "".join(re.escape(d) for d in DIVIDERS) + "]\s+)(.*?)$"
        )
        match = year_pattern.match(text)

        if not match:
            print(f"**** Failed to parse: {text}")
            continue

        year, result = match.groups()

        # Clean up the text
        result = re.sub(r"\[\d+\]", "", result).strip()
        if not result:
            print(f"**** Empty result: {text}")
            continue

        # Remove any superscripts
        for sup in item.find_all("sup"):
            sup.decompose()

        # Process links
        links = []
        for link in item.find_all("a"):
            href = link.get("href", "")
            if href.startswith("/"):
                link["href"] = f"{url_base}{href}"

            title = link.get("title")
            if title:
                links.append({"title": title, "link": link["href"]})

        # Clean up the year
        year = year.replace("AD ", "").replace("BC ", "").strip()

        # Get the HTML without the year prefix
        html_content = str(item)
        for divider in DIVIDERS:
            pattern = re.compile(
                f"^{re.escape(year)}\\s*{re.escape(divider)}\\s*", re.DOTALL
            )
            html_content = pattern.sub("", html_content)

        # Remove any leading divider
        html_content = re.sub(r"^\s*[–-]\s*", "", html_content)

        data.append(
            {
                "year": year,
                "text": result,
                "html": f"{year} - {html_content}",
                "no_year_html": html_content,
                "links": links,
            }
        )

    return data


def validate_data(data):
    """Validate the extracted data meets our requirements"""
    # Check for required base fields
    missing = [field for field in EXPECTED_BASE if field not in data]
    if missing:
        raise ValueError(f"Missing base data: {missing}")

    # Check for required data sections
    missing = [field for field in EXPECTED_DATA if field not in data["data"]]
    if missing:
        raise ValueError(f"Missing data sections: {missing}")

    # Check for recent events
    years = [
        int(entry["year"])
        for entry in data["data"]["Events"]
        if entry["year"].isdigit()
    ]

    if not years or max(years) < date.today().year - MIN_YEAR:
        raise ValueError(f"No events in the last {MIN_YEAR} years")

    # Check for minimum number of events
    event_count = len(data["data"]["Events"])
    if event_count < MIN_EVENTS:
        raise ValueError(
            f"Only {event_count} events found, minimum required is {MIN_EVENTS}"
        )


def main():
    """Main execution function"""
    # Process each day of the year (using 2000 as a leap year to get Feb 29)
    start_date = date(2000, 1, 1)
    end_date = date(2000, 12, 31)

    current_date = start_date
    while current_date <= end_date:
        actual_date = current_date.strftime("%m-%d")
        wiki_date = current_date.strftime("%B %d").replace(" 0", " ")

        print(f"Processing: {wiki_date}")

        # Data structure to store results
        data = {}

        # Check for cached HTML file
        cached_file = os.path.join(DEST, f"{actual_date}.html")

        if os.path.exists(cached_file):
            with open(cached_file, "r", encoding="utf-8") as f:
                html_content = f.read()
        else:
            # Construct Wikipedia URL
            url = f"https://en.wikipedia.org/wiki/{current_date.strftime('%B')}_{current_date.strftime('%d').strip()}"
            print(f"Fetching: {url}")

            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching {url}: {response.status_code}")
                current_date += timedelta(days=1)
                continue

            html_content = response.text

            # Cache HTML if requested
            if CACHE_HTML:
                with open(cached_file, "w", encoding="utf-8") as f:
                    f.write(html_content)

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Clean up the HTML
        for elem in soup.select("div.thumb, .mw-editsection"):
            elem.decompose()

        # Process each section
        sectional_data = {}

        # Find all unordered lists
        for ul in soup.find_all("ul"):
            header = closest_header(ul)
            if not header:
                continue

            section_title = header.get_text().strip()
            if section_title not in SECTIONS:
                continue

            if section_title not in sectional_data:
                sectional_data[section_title] = []

            sectional_data[section_title].extend(process_list(ul))

        # Construct final result
        result = {
            "date": wiki_date,
            "url": f"https://wikipedia.org/wiki/{wiki_date.replace(' ', '_')}",
            "data": sectional_data,
        }

        try:
            validate_data(result)

            # Save as JSON
            output_file = os.path.join(DEST, f"{actual_date}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"Saved: {output_file}")

        except ValueError as e:
            print(f"Validation error for {wiki_date}: {e}")

        # Move to next day
        current_date += timedelta(days=1)


if __name__ == "__main__":
    main()
