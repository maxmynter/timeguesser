# /// script
# requires-python = ">=3.13"
# dependencies = [
#    "matplotlib",
#    "numpy"
#    ]
# ///


import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter, defaultdict

# Input file containing all events
INPUT_FILE = "historical_events.json"


def load_events():
    """Load all historical events from the unified JSON file"""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["events"]


def analyze_years(events):
    """Analyze the distribution of events by year"""
    # Extract years from events, filtering out None values
    years = [event["year"] for event in events if event["year"] is not None]

    # Count events per year
    year_counts = Counter(years)

    # Get the range of years
    min_year = min(years)
    max_year = max(years)

    print(f"Year range: {min_year} to {max_year}")
    print(f"Total events with valid years: {len(years)}")

    return year_counts, min_year, max_year


def analyze_by_month_day(events):
    """Analyze the distribution of events by month and day"""
    month_counts = Counter()
    day_counts = defaultdict(Counter)

    for event in events:
        month = event["month"]
        day = event["day"]
        month_counts[month] += 1
        day_counts[month][day] += 1

    return month_counts, day_counts


def analyze_by_type(events):
    """Analyze the distribution of events by type"""
    type_counts = Counter()
    type_by_year = defaultdict(Counter)

    for event in events:
        event_type = event["type"]
        year = event["year"]
        type_counts[event_type] += 1

        if year is not None:
            # Group years into centuries for this analysis
            century = (year // 100) * 100
            type_by_year[event_type][century] += 1

    return type_counts, type_by_year


def plot_histogram(events):
    """Plot a histogram of events by year"""
    year_counts, min_year, max_year = analyze_years(events)

    # Create bins for the histogram - adjust these parameters based on your data
    if min_year < 0 and max_year > 0:
        # Handle BC/BCE and CE/AD years with different bin sizes
        bins_bc = np.linspace(min_year, -1, 50)
        bins_ad = np.linspace(0, max_year, 100)
        bins = np.concatenate([bins_bc, bins_ad])
    else:
        # Regular binning
        bins = 100

    # Extract years and counts for plotting
    years = sorted(year_counts.keys())
    counts = [year_counts[year] for year in years]

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Main histogram
    plt.hist(years, bins=bins, weights=counts, alpha=0.7, color="blue")

    # Add title and labels
    plt.title("Distribution of Historical Events by Year", fontsize=16)
    plt.xlabel("Year (negative values are BCE/BC)", fontsize=14)
    plt.ylabel("Number of Events", fontsize=14)

    # Add a grid
    plt.grid(True, alpha=0.3)

    # Adjust axis for better visibility
    if min_year < 0:
        plt.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, label="Year 0 (BC/AD divide)"
        )
        plt.legend()

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("events_histogram.png", dpi=300)
    print("Saved histogram to events_histogram.png")

    # Close the plot to free memory
    plt.close()


def plot_recent_years(events, start_year=1500):
    """Plot a histogram focused on more recent years"""
    year_counts, _, _ = analyze_years(events)

    # Filter to only show events after start_year
    recent_years = {
        year: count for year, count in year_counts.items() if year >= start_year
    }

    if not recent_years:
        print(f"No events found after year {start_year}")
        return

    # Extract years and counts for plotting
    years = sorted(recent_years.keys())
    counts = [recent_years[year] for year in years]

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Bar plot for recent years
    plt.bar(years, counts, alpha=0.7, color="green")

    # Add title and labels
    plt.title(
        f"Distribution of Historical Events from {start_year} to Present", fontsize=16
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of Events", fontsize=14)

    # Add a grid
    plt.grid(True, alpha=0.3)

    # Improve x-axis
    plt.xticks(rotation=45)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("recent_events_histogram.png", dpi=300)
    print(
        f"Saved recent years histogram (from {start_year}) to recent_events_histogram.png"
    )

    # Close the plot to free memory
    plt.close()


def plot_by_type(events):
    """Plot events by type (event, birth, death)"""
    type_counts, type_by_year = analyze_by_type(events)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Bar plot for types
    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]

    plt.bar(types, counts, alpha=0.7, color=["blue", "green", "red"])

    # Add title and labels
    plt.title("Distribution of Historical Records by Type", fontsize=16)
    plt.xlabel("Type", fontsize=14)
    plt.ylabel("Number of Records", fontsize=14)

    # Add counts on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + 100, f"{count}", ha="center", fontsize=12)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("events_by_type.png", dpi=300)
    print("Saved type distribution to events_by_type.png")

    # Close the plot to free memory
    plt.close()


def plot_heatmap(events):
    """Plot a heatmap of events by month and day"""
    _, day_counts = analyze_by_month_day(events)

    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((12, 31))

    for month in range(1, 13):
        for day in range(1, 32):
            if day in day_counts[month]:
                heatmap_data[month - 1, day - 1] = day_counts[month][day]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot heatmap
    plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
    plt.colorbar(label="Number of Events")

    # Add title and labels
    plt.title("Distribution of Events by Month and Day", fontsize=16)
    plt.xlabel("Day of Month", fontsize=14)
    plt.ylabel("Month", fontsize=14)

    # Set ticks
    plt.yticks(
        np.arange(12),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    plt.xticks(np.arange(0, 31, 5), np.arange(1, 32, 5))

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("events_heatmap.png", dpi=300)
    print("Saved month/day heatmap to events_heatmap.png")

    # Close the plot to free memory
    plt.close()


def main():
    """Main function to generate all plots"""
    if not os.path.exists(INPUT_FILE):
        print(
            f"Error: {INPUT_FILE} not found. Please run the data processing script first."
        )
        return

    try:
        print(f"Loading events from {INPUT_FILE}...")
        events = load_events()
        print(f"Loaded {len(events)} events")

        # Generate various plots
        plot_histogram(events)
        plot_recent_years(events, start_year=1500)
        plot_by_type(events)
        plot_heatmap(events)

        print("All plots generated successfully!")

    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == "__main__":
    main()
