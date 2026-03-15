"""Pinnacle steam monitor: detect sharp line movements and alert for AU book bets.

The thesis: Pinnacle is the sharpest UFC book. When their line moves significantly,
AU books (Sportsbet, TAB, Ladbrokes) lag behind. We bet the AU books before they
adjust.

Usage:
    python steam_monitor.py --poll-interval 300

Requires ODDS_API_KEY in .env for live odds polling.
"""

import argparse
import time
from datetime import datetime

import requests

from config import (
    ODDS_API_KEY,
    PINNACLE_POLL_INTERVAL,
    STEAM_THRESHOLD,
)


# Bookmaker groupings
SHARP_BOOKS = {"pinnacle"}
AU_BOOKS = {"sportsbet", "tab", "pointsbetau", "unibet", "ladbrokes_au", "neds", "betfair_ex_au"}


def fetch_ufc_odds() -> list:
    """Fetch current UFC odds from The Odds API."""
    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not set in .env")
        return []

    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "au,us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def detect_steam(prev_odds: dict, curr_odds: dict) -> list:
    """Compare two snapshots of odds and detect significant Pinnacle movements.

    Returns list of steam alerts with the AU book opportunities.
    """
    alerts = []

    for event_id, curr_event in curr_odds.items():
        prev_event = prev_odds.get(event_id)
        if not prev_event:
            continue

        pinnacle_prev = prev_event.get("pinnacle")
        pinnacle_curr = curr_event.get("pinnacle")
        if not pinnacle_prev or not pinnacle_curr:
            continue

        for fighter in pinnacle_curr:
            prev_odds_val = pinnacle_prev.get(fighter)
            curr_odds_val = pinnacle_curr[fighter]
            if not prev_odds_val:
                continue

            prev_implied = 1.0 / prev_odds_val
            curr_implied = 1.0 / curr_odds_val
            move = curr_implied - prev_implied

            if abs(move) >= STEAM_THRESHOLD:
                # Find AU books that haven't adjusted yet
                au_opps = []
                for book_name in AU_BOOKS:
                    au_odds = curr_event.get(book_name, {}).get(fighter)
                    if au_odds:
                        au_implied = 1.0 / au_odds
                        lag = curr_implied - au_implied
                        if lag > 0.01:  # AU book is still at old price
                            au_opps.append({
                                "book": book_name,
                                "odds": au_odds,
                                "implied": au_implied,
                                "lag": lag,
                            })

                if au_opps:
                    alerts.append({
                        "event_id": event_id,
                        "fighter": fighter,
                        "direction": "shorter" if move > 0 else "longer",
                        "pinnacle_move": move,
                        "pinnacle_odds": curr_odds_val,
                        "au_opportunities": au_opps,
                    })

    return alerts


def _parse_snapshot(events: list) -> dict:
    """Parse API response into {event_id: {bookmaker: {fighter: odds}}}."""
    snapshot = {}
    for event in events:
        event_id = event["id"]
        books = {}
        for bookmaker in event.get("bookmakers", []):
            key = bookmaker["key"]
            market = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
            if market:
                books[key] = {
                    o["name"]: o["price"] for o in market["outcomes"]
                }
        snapshot[event_id] = books
    return snapshot


def run_monitor(poll_interval: int = PINNACLE_POLL_INTERVAL):
    """Main monitoring loop."""
    print(f"Steam monitor started. Polling every {poll_interval}s.")
    print(f"Steam threshold: {STEAM_THRESHOLD:.0%} implied prob move")
    print(f"Watching sharp books: {SHARP_BOOKS}")
    print(f"Target AU books: {AU_BOOKS}")
    print()

    prev_snapshot = None

    while True:
        try:
            events = fetch_ufc_odds()
            curr_snapshot = _parse_snapshot(events)

            if prev_snapshot:
                alerts = detect_steam(prev_snapshot, curr_snapshot)
                for alert in alerts:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] STEAM: {alert['fighter']} moved {alert['direction']} "
                          f"(Pinnacle {alert['pinnacle_move']:+.3f})")
                    for opp in alert["au_opportunities"]:
                        print(f"  -> {opp['book']}: {opp['odds']:.2f} "
                              f"(lag: {opp['lag']:.3f} implied)")

            prev_snapshot = curr_snapshot
            n_events = len(events)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Polled {n_events} events")

        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pinnacle Steam Monitor for UFC")
    parser.add_argument("--poll-interval", type=int, default=PINNACLE_POLL_INTERVAL)
    args = parser.parse_args()
    run_monitor(args.poll_interval)
