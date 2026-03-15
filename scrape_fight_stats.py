"""Scrape per-fight stats from ufcstats.com for point-in-time feature engineering.

Scrapes the totals table from each fight detail page:
  KD, Sig. str. (landed/attempted), Total str., Td (landed/attempted),
  Sub. att, Rev., Ctrl (control time in seconds)

Output: data/fight_stats.parquet with one row per fighter per fight.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "fight_stats.parquet")

BASE_URL = "http://ufcstats.com"
EVENTS_URL = f"{BASE_URL}/statistics/events/completed?page=all"
MAX_WORKERS = 8

session = requests.Session()


def _get_soup(url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_all_events() -> list[dict]:
    """Get all event URLs and names."""
    soup = _get_soup(EVENTS_URL)
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


def get_fight_links(event_url: str) -> list[str]:
    """Get fight detail URLs from an event page."""
    resp = session.get(event_url, timeout=30)
    resp.raise_for_status()
    return list(set(re.findall(
        r'data-link="(http://ufcstats\.com/fight-details/[a-f0-9]+)"',
        resp.text
    )))


def _parse_of(text: str) -> tuple[int, int]:
    m = re.match(r"(\d+)\s+of\s+(\d+)", text.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _parse_ctrl(text: str) -> int:
    text = text.strip()
    if not text or text == "---":
        return 0
    parts = text.split(":")
    return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0


def parse_fight_totals(fight_url: str) -> list[dict] | None:
    """Parse the totals table from a fight detail page."""
    soup = _get_soup(fight_url)
    tables = soup.find_all("table")
    if not tables:
        return None

    table = tables[0]
    thead = table.find("thead")
    if not thead:
        return None

    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    if "KD" not in headers:
        return None

    tbody = table.find("tbody")
    if not tbody:
        return None

    rows = tbody.find_all("tr")
    if not rows:
        return None

    row = rows[0]
    cells = row.find_all("td")
    fighter_cell = cells[0]
    fighter_ps = fighter_cell.find_all("p")
    if len(fighter_ps) < 2:
        return None

    fighters = []
    for p in fighter_ps[:2]:
        link = p.find("a")
        fighters.append(link.get_text(strip=True) if link else p.get_text(strip=True))

    results = []
    for i, fighter in enumerate(fighters):
        stat = {"fighter": fighter, "fight_url": fight_url}

        for col_idx, header in enumerate(headers[1:], 1):
            if col_idx >= len(cells):
                break
            cell = cells[col_idx]
            ps = cell.find_all("p")
            if len(ps) < 2:
                continue
            val = ps[i].get_text(strip=True)

            if header == "KD":
                stat["knockdowns"] = int(val) if val.isdigit() else 0
            elif header == "Sig. str.":
                landed, attempted = _parse_of(val)
                stat["sig_str_landed"] = landed
                stat["sig_str_attempted"] = attempted
            elif header == "Total str.":
                landed, attempted = _parse_of(val)
                stat["total_str_landed"] = landed
                stat["total_str_attempted"] = attempted
            elif header == "Td":
                landed, attempted = _parse_of(val)
                stat["td_landed"] = landed
                stat["td_attempted"] = attempted
            elif header == "Sub. att":
                stat["sub_att"] = int(val) if val.isdigit() else 0
            elif header == "Rev.":
                stat["reversals"] = int(val) if val.isdigit() else 0
            elif header == "Ctrl":
                stat["ctrl_seconds"] = _parse_ctrl(val)

        results.append(stat)

    return results


def _scrape_event(event: dict) -> list[dict]:
    """Scrape all fights from one event. Returns list of stat dicts."""
    try:
        fight_links = get_fight_links(event["url"])
    except Exception:
        return []

    stats = []
    for fight_url in fight_links:
        try:
            result = parse_fight_totals(fight_url)
            if result:
                for s in result:
                    s["event_name"] = event["name"]
                    s["event_date"] = event["date"]
                stats.extend(result)
        except Exception:
            pass
    return stats


def scrape_all():
    """Scrape all fight stats from ufcstats.com using parallel requests."""
    print("Getting event list...", flush=True)
    events = get_all_events()
    print(f"  {len(events)} events found", flush=True)

    all_stats = []
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_scrape_event, ev): ev for ev in events}
        for future in as_completed(futures):
            stats = future.result()
            all_stats.extend(stats)
            done += 1
            if done % 50 == 0:
                fights = len(all_stats) // 2
                print(f"  {done}/{len(events)} events, ~{fights} fights scraped", flush=True)

    fights = len(all_stats) // 2
    print(f"\nDone: ~{fights} fights scraped", flush=True)

    df = pd.DataFrame(all_stats)
    df["event_date"] = pd.to_datetime(df["event_date"], format="mixed", errors="coerce")
    df = df.sort_values("event_date").reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    scrape_all()
