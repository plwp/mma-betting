"""Scrape fight results from ufcstats.com event pages.

Extends fights.parquet with results from Oct 2023 onwards.
Also scrapes historical odds from BestFightOdds where available.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import DATA_DIR, FIGHTS_PATH

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
})


def get_all_events():
    """Get all event URLs from ufcstats."""
    resp = session.get(
        "http://ufcstats.com/statistics/events/completed?page=all", timeout=30
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    events = []
    seen = set()
    for link in soup.find_all("a", href=re.compile(r"event-details")):
        url = link["href"].strip()
        if url in seen:
            continue
        seen.add(url)
        row = link.find_parent("tr")
        date_cell = row.find("span", class_="b-statistics__date") if row else None
        events.append({
            "url": url,
            "name": link.get_text(strip=True),
            "date": date_cell.get_text(strip=True) if date_cell else "",
        })
    return events


def scrape_event_results(event_url: str) -> list[dict]:
    """Scrape fight results from an event page."""
    resp = session.get(event_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Get event name and date
    title = soup.find("h2", class_="b-content__title")
    event_name = title.get_text(strip=True) if title else ""

    date_items = soup.find_all("li", class_="b-list__box-list-item")
    event_date = ""
    for item in date_items:
        text = item.get_text(strip=True)
        if text.startswith("Date:"):
            event_date = text.replace("Date:", "").strip()
            break

    fights = []
    rows = soup.find_all("tr", class_="b-fight-details__table-row")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        # Parse cells
        cell_data = []
        for cell in cells:
            ps = cell.find_all("p")
            if ps:
                cell_data.append([p.get_text(strip=True) for p in ps])
            else:
                cell_data.append([cell.get_text(strip=True)])

        # Structure: [outcome, fighters, KD, STR, TD, SUB, weight_class, method, round, time]
        if len(cell_data) < 10:
            continue

        outcome = cell_data[0][0].strip().lower() if cell_data[0] else ""
        fighters = cell_data[1] if len(cell_data[1]) >= 2 else None
        if not fighters:
            continue

        fighter1 = fighters[0].strip()
        fighter2 = fighters[1].strip()
        weight_class = cell_data[6][0].strip() if cell_data[6] else ""
        method_parts = cell_data[7] if len(cell_data[7]) >= 1 else [""]
        method = method_parts[0].strip()
        fight_round = cell_data[8][0].strip() if cell_data[8] else ""
        fight_time = cell_data[9][0].strip() if cell_data[9] else ""

        if outcome == "win":
            winner = fighter1
        elif outcome == "draw" or outcome == "nc":
            winner = None  # Skip draws/NCs
        else:
            continue

        if winner is None:
            continue

        fights.append({
            "event_name": event_name,
            "event_date": event_date,
            "weight_class": weight_class,
            "fighter1": fighter1,  # winner
            "fighter2": fighter2,  # loser
            "winner": winner,
            "method": method,
            "round": fight_round,
            "time": fight_time,
        })

    return fights


def scrape_bfo_event_odds(event_name: str) -> dict:
    """Try to scrape odds from BestFightOdds for an event.

    Returns {(fighter1, fighter2): (odds1, odds2)} or empty dict.
    """
    # BestFightOdds uses URL slugs — try to find the event
    # This is best-effort; BFO may block or not have the event
    try:
        search_url = f"https://www.bestfightodds.com/search?query={event_name.replace(' ', '+')}"
        resp = session.get(search_url, timeout=15)
        if resp.status_code != 200:
            return {}
        # TODO: parse BFO search results for event odds
        # For now, return empty — we'll add BFO scraping later
        return {}
    except Exception:
        return {}


def extend_dataset():
    """Scrape results for events after our current dataset and extend fights.parquet."""
    # Load existing data
    existing = pd.read_parquet(FIGHTS_PATH)
    last_date = existing["date"].max()
    print(f"Existing data ends: {last_date.date()}")

    # Get all events
    print("Getting event list...")
    events = get_all_events()
    print(f"  {len(events)} total events")

    # Filter to events after our data
    new_events = []
    for ev in events:
        try:
            ev_date = pd.to_datetime(ev["date"])
            if ev_date > last_date:
                new_events.append(ev)
        except Exception:
            continue

    print(f"  {len(new_events)} events after {last_date.date()}")

    # Scrape results
    all_fights = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scrape_event_results, ev["url"]): ev for ev in new_events}
        done = 0
        for future in as_completed(futures):
            fights = future.result()
            all_fights.extend(fights)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(new_events)} events scraped, {len(all_fights)} fights", flush=True)

    print(f"\nScraped {len(all_fights)} new fights")

    if not all_fights:
        print("No new fights found.")
        return existing

    # Convert to DataFrame matching existing format
    new_df = pd.DataFrame(all_fights)
    new_df["date"] = pd.to_datetime(new_df["event_date"], format="mixed", errors="coerce")
    new_df["year"] = new_df["date"].dt.year

    # Randomize fighter ordering (same approach as data_ingest.py)
    rng = np.random.RandomState(42)
    swap = rng.random(len(new_df)) > 0.5
    new_df["fighter_a"] = np.where(swap, new_df["fighter2"], new_df["fighter1"])
    new_df["fighter_b"] = np.where(swap, new_df["fighter1"], new_df["fighter2"])
    new_df["a_wins"] = np.where(
        new_df["winner"] == new_df["fighter_a"], 1, 0
    ).astype(int)

    # Method category
    new_df["method_cat"] = new_df["method"].str.extract(
        r"^(KO/TKO|SUB|U-DEC|S-DEC|M-DEC|DQ|Overturned)"
    )
    new_df["method_cat"] = new_df["method_cat"].fillna("Other")

    # Context features
    new_df["is_title_fight"] = new_df["event_name"].str.contains(
        r"title|championship|interim", case=False, na=False
    ).astype(int)
    new_df["is_main_event"] = 0

    # No odds yet — will be NaN
    new_df["odds_a"] = np.nan
    new_df["odds_b"] = np.nan
    new_df["market_prob"] = np.nan
    new_df["market_overround"] = np.nan

    # Physical attributes — fill with NaN (will be filled from existing data later)
    for col in ["height_diff_cm", "reach_diff_cm", "age_diff", "age_fighter",
                "age_a", "age_b", "stance_matchup", "weight_class_encoded"]:
        if col not in new_df.columns:
            new_df[col] = np.nan

    # Weight class encoding
    from config import WEIGHT_CLASSES
    new_df["weight_class_encoded"] = new_df["weight_class"].map(WEIGHT_CLASSES).fillna(170)

    # Keep columns that exist in both
    keep_cols = [c for c in existing.columns if c in new_df.columns]
    # Add any columns from new_df that are in existing but might be missing
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    # Combine
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Fix mixed types from concat
    for col in combined.columns:
        if combined[col].dtype == object:
            combined[col] = pd.to_numeric(combined[col], errors="ignore")

    # Deduplicate by fighter pair + date
    combined = combined.drop_duplicates(
        subset=["date", "fighter_a", "fighter_b"], keep="first"
    ).reset_index(drop=True)

    print(f"\nCombined: {len(combined)} fights ({len(combined) - len(existing)} new)")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"With odds: {combined['odds_a'].notna().sum()}")
    print(f"a_wins mean: {combined['a_wins'].mean():.3f}")

    combined.to_parquet(FIGHTS_PATH, index=False)
    print(f"Saved to {FIGHTS_PATH}")

    return combined


if __name__ == "__main__":
    extend_dataset()
