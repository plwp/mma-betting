"""Download and merge UFC fight data with historical odds.

Data sources:
- ufcstats.com: fight results, round-by-round stats
- BestFightOdds: historical closing odds from 12+ bookmakers

Uses the `ufcscraper` package for structured scraping.
Install: pip install ufcscraper
"""

import os
import subprocess
import sys

import pandas as pd

from config import DATA_DIR, FIGHTS_PATH, ODDS_PATH


def _ensure_ufcscraper():
    """Check if ufcscraper is installed."""
    try:
        import ufcscraper  # noqa: F401
        return True
    except ImportError:
        print("ufcscraper not installed. Install with: pip install ufcscraper")
        return False


def scrape_fight_stats(output_dir: str = None) -> pd.DataFrame:
    """Scrape fight stats from ufcstats.com using ufcscraper."""
    output_dir = output_dir or os.path.join(DATA_DIR, "raw_stats")
    os.makedirs(output_dir, exist_ok=True)

    print("Scraping UFC fight stats from ufcstats.com...")
    subprocess.run(
        [sys.executable, "-m", "ufcscraper", "scrape-ufcstats", "--output-dir", output_dir],
        check=True,
    )

    # Load the scraped data
    fights_file = os.path.join(output_dir, "fights.csv")
    if not os.path.exists(fights_file):
        raise FileNotFoundError(f"Expected {fights_file} after scraping")

    return pd.read_csv(fights_file)


def scrape_odds(output_dir: str = None) -> pd.DataFrame:
    """Scrape historical odds from BestFightOdds using ufcscraper."""
    output_dir = output_dir or os.path.join(DATA_DIR, "raw_odds")
    os.makedirs(output_dir, exist_ok=True)

    print("Scraping odds from BestFightOdds...")
    subprocess.run(
        [sys.executable, "-m", "ufcscraper", "scrape-bestfightodds", "--output-dir", output_dir],
        check=True,
    )

    odds_file = os.path.join(output_dir, "odds.csv")
    if not os.path.exists(odds_file):
        raise FileNotFoundError(f"Expected {odds_file} after scraping")

    return pd.read_csv(odds_file)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    elif american_odds < 0:
        return 1 + (100 / abs(american_odds))
    return 1.0


def merge_fights_and_odds(fights: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Merge fight results with odds data.

    This is a placeholder -- the exact merge logic depends on the ufcscraper
    output format. Will need to be adjusted after first scrape.
    """
    # TODO: Implement after inspecting actual ufcscraper output format
    # Key steps:
    # 1. Normalize fighter names across both datasets
    # 2. Match by event date + fighter names
    # 3. Convert American odds to decimal
    # 4. Compute implied probabilities and overround
    print(f"Fights shape: {fights.shape}")
    print(f"Odds shape: {odds.shape}")
    print(f"Fights columns: {list(fights.columns)}")
    print(f"Odds columns: {list(odds.columns)}")
    return fights


def run():
    """Full data ingestion pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not _ensure_ufcscraper():
        print("\nAlternative: download pre-built datasets from Kaggle:")
        print("  - https://www.kaggle.com/datasets/jerzyszocik/ufc-betting-odds-daily-dataset")
        print("  - https://github.com/jansen88/ufc-data")
        return None

    fights = scrape_fight_stats()
    odds = scrape_odds()
    merged = merge_fights_and_odds(fights, odds)

    merged.to_parquet(FIGHTS_PATH, index=False)
    print(f"Saved {len(merged)} fights to {FIGHTS_PATH}")
    return merged


if __name__ == "__main__":
    run()
