"""
MBTA Bus Delay Heatmap (2025)
==============================
Generates an interactive Folium heatmap showing average bus delay
across Boston stops by time of day, using the MBTA Arrival/Departure
Times 2025 dataset.

Requirements:
    pip install pandas folium requests

Usage:
    python mbta_delay_heatmap.py --csv your_arrival_departure_file.csv
    python mbta_delay_heatmap.py --csv your_file.csv --output my_map.html
"""

import argparse
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime
import requests

# ── 1. LOAD & PARSE DATA ──────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")
    return df


def compute_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delay in minutes as actual - scheduled.
    Handles overnight wraparound (e.g. scheduled=11:55 PM, actual=12:03 AM).
    Only uses rows where standard_type == 'Schedule' for delay analysis.
    """
    # Work on schedule-standard rows only (headway rows don't use scheduled time)
    sched_mask = df["standard_type"].str.lower().str.contains("schedule", na=False)
    df = df[sched_mask].copy()

    # Parse times — MBTA uses HH:MM AM/PM format; some times exceed 23:59 (e.g. 25:30)
    # so we use a safe parser
    def parse_time_to_minutes(series: pd.Series) -> pd.Series:
        """Convert time strings like '12:30 AM' or '25:15' to minutes since midnight."""
        def _parse(t):
            if pd.isna(t):
                return None
            t = str(t).strip()
            try:
                # Standard 12-hour format
                from datetime import datetime
                dt = datetime.strptime(t, "%I:%M %p")
                return dt.hour * 60 + dt.minute
            except ValueError:
                pass
            try:
                # 24-hour or extended (e.g. 25:30 for 1:30 AM next day)
                parts = t.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return None

        return series.apply(_parse)

    df["scheduled_min"] = parse_time_to_minutes(df["scheduled"])
    df["actual_min"]    = parse_time_to_minutes(df["actual"])
    df = df.dropna(subset=["scheduled_min", "actual_min"])

    # Delay in minutes; handle midnight wraparound
    df["delay_min"] = df["actual_min"] - df["scheduled_min"]
    # If bus appears to be > 6 hours early it's probably a wraparound
    df.loc[df["delay_min"] < -360, "delay_min"] += 1440
    df.loc[df["delay_min"] >  360, "delay_min"] -= 1440

    print(f"Delay stats (minutes):\n{df['delay_min'].describe().round(2)}\n")
    return df


def assign_time_period(df: pd.DataFrame) -> pd.DataFrame:
    """Bin actual departure minute into named time periods."""
    def _period(m):
        if m is None:
            return "UNKNOWN"
        m = m % 1440  # normalise extended hours
        if   m <  5*60:            return "VERY_EARLY_MORNING"
        elif m <  9*60:            return "AM_PEAK"
        elif m < 12*60:            return "MIDDAY_AM"
        elif m < 14*60:            return "MIDDAY_PM"
        elif m < 18*60:            return "PM_PEAK"
        elif m < 21*60:            return "EVENING"
        else:                      return "LATE_NIGHT"

    df["time_period"] = df["actual_min"].apply(_period)
    return df


# ── 2. FETCH MBTA STOP COORDINATES ───────────────────────────────────────────

def fetch_mbta_stop_coords() -> dict:
    """Returns {stop_id (str): (lat, lon)} for all bus stops."""
    coords = {}
    url = "https://api-v3.mbta.com/stops?filter[route_type]=3&per_page=500"
    while url:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for stop in data.get("data", []):
            lat = stop["attributes"]["latitude"]
            lon = stop["attributes"]["longitude"]
            if lat and lon:
                coords[stop["id"]] = (lat, lon)
        url = data.get("links", {}).get("next")
    print(f"Fetched coordinates for {len(coords):,} MBTA bus stops.")
    return coords


def merge_coords(df: pd.DataFrame, coords: dict) -> pd.DataFrame:
    df["stop_id"] = df["stop_id"].astype(str)
    df["lat"] = df["stop_id"].map(lambda s: coords.get(s, (None, None))[0])
    df["lon"] = df["stop_id"].map(lambda s: coords.get(s, (None, None))[1])
    missing = df["lat"].isna().sum()
    print(f"Matched {len(df)-missing:,} rows; {missing:,} stops unmatched.")
    return df.dropna(subset=["lat", "lon"])


# ── 3. BUILD HEATMAP ──────────────────────────────────────────────────────────

TIME_PERIOD_ORDER = [
    "VERY_EARLY_MORNING",
    "AM_PEAK",
    "MIDDAY_AM",
    "MIDDAY_PM",
    "PM_PEAK",
    "EVENING",
    "LATE_NIGHT",
]

def build_heatmap(df: pd.DataFrame, output_path: str):
    # Aggregate average delay per stop per time period
    agg = (
        df.groupby(["time_period", "stop_id", "lat", "lon"])["delay_min"]
        .mean()
        .reset_index()
        .rename(columns={"delay_min": "avg_delay"})
    )

    # Clip extreme outliers for colour scaling (cap at 15 min late)
    agg["avg_delay_clipped"] = agg["avg_delay"].clip(lower=0, upper=15)
    max_delay = agg["avg_delay_clipped"].max()
    if max_delay == 0:
        max_delay = 1
    agg["intensity"] = agg["avg_delay_clipped"] / max_delay

    present = [p for p in TIME_PERIOD_ORDER if p in agg["time_period"].unique()]
    extra   = [p for p in agg["time_period"].unique() if p not in TIME_PERIOD_ORDER]
    all_periods = present + extra

    heat_data = []
    for period in all_periods:
        subset = agg[agg["time_period"] == period]
        heat_data.append(subset[["lat", "lon", "intensity"]].values.tolist())

    # Base map
    m = folium.Map(location=[42.3601, -71.0589], zoom_start=12, tiles="CartoDB positron")

    HeatMapWithTime(
        data=heat_data,
        index=all_periods,
        name="Bus Delay",
        radius=20,
        min_opacity=0.3,
        max_opacity=0.9,
        gradient={0.0: "green", 0.4: "yellow", 0.7: "orange", 1.0: "red"},
        use_local_extrema=False,
        auto_play=False,
        display_index=True,
    ).add_to(m)

    # Stop markers with delay tooltip
    stop_layer = folium.FeatureGroup(name="Stop Details (hover)", show=False)
    for _, row in agg.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="steelblue",
            fill=True,
            fill_opacity=0.5,
            tooltip=(
                f"Stop {row['stop_id']}<br>"
                f"Period: {row['time_period']}<br>"
                f"Avg Delay: {row['avg_delay']:.1f} min"
            ),
        ).add_to(stop_layer)
    stop_layer.add_to(m)
    folium.LayerControl().add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.3);font-size:13px;">
      <b>Average Bus Delay</b><br>
      <span style="color:green">&#9632;</span> On time &nbsp;
      <span style="color:yellow">&#9632;</span> Slight delay &nbsp;
      <span style="color:orange">&#9632;</span> Moderate &nbsp;
      <span style="color:red">&#9632;</span> Severe (&ge;15 min)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    print(f"\n✅ Heatmap saved to: {output_path}")
    print("Open it in any browser to explore delays by time period.")


# ── 4. MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MBTA Bus Delay Heatmap")
    parser.add_argument("--csv",    required=True, help="Path to arrival/departure CSV")
    parser.add_argument("--output", default="mbta_delay_heatmap.html", help="Output HTML file")
    args = parser.parse_args()

    df = load_data(args.csv)
    df = compute_delay(df)
    df = assign_time_period(df)

    coords = fetch_mbta_stop_coords()
    df = merge_coords(df, coords)

    build_heatmap(df, output_path=args.output)


if __name__ == "__main__":
    main()