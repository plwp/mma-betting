"""Download and process UFC fight data with historical odds.

Primary source: jansen88/ufc-data (GitHub) — pre-built CSV with 30 years of
fight history + 9 years of odds from ufcstats.com and betmma.tips.

CRITICAL: The source data has fighter1=winner always. We randomize fighter
ordering to prevent label leakage.
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, FIGHTS_PATH


RAW_CSV = os.path.join(DATA_DIR, "complete_ufc_data.csv")
RAW_URL = (
    "https://raw.githubusercontent.com/jansen88/ufc-data/"
    "master/data/complete_ufc_data.csv"
)


def download_raw_data() -> pd.DataFrame:
    """Download the raw CSV if not cached."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(RAW_CSV):
        print(f"Downloading from {RAW_URL}...")
        import requests
        resp = requests.get(RAW_URL, timeout=60)
        resp.raise_for_status()
        with open(RAW_CSV, "wb") as f:
            f.write(resp.content)

    return pd.read_csv(RAW_CSV)


def process_fights(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and restructure fight data.

    The raw data always has fighter1=winner. We randomize the ordering
    with a deterministic seed to prevent leakage while keeping results
    reproducible.
    """
    df = raw.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["event_date"])
    df["year"] = df["date"].dt.year

    # Drop draws and no-contests (can't bet on these)
    df = df[df["outcome"] == "fighter1"].copy()
    print(f"  Dropped {len(raw) - len(df)} draws/NCs, {len(df)} fights remain")

    # Clean infinite odds
    for col in ["favourite_odds", "underdog_odds"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Randomize fighter ordering to prevent leakage
    # fighter1 is ALWAYS the winner in the raw data
    rng = np.random.RandomState(42)
    swap = rng.random(len(df)) > 0.5

    # Create new columns with randomized ordering
    df["fighter_a"] = np.where(swap, df["fighter2"], df["fighter1"])
    df["fighter_b"] = np.where(swap, df["fighter1"], df["fighter2"])
    df["a_wins"] = (~swap).astype(int)  # 1 if fighter_a wins, 0 if fighter_b wins

    # Swap physical/stats columns accordingly
    stat_suffixes = [
        "height", "curr_weight", "dob", "reach", "stance",
        "sig_strikes_landed_pm", "sig_strikes_accuracy",
        "sig_strikes_absorbed_pm", "sig_strikes_defended",
        "takedown_avg_per15m", "takedown_accuracy",
        "takedown_defence", "submission_avg_attempted_per15m",
    ]

    for suffix in stat_suffixes:
        col1 = f"fighter1_{suffix}"
        col2 = f"fighter2_{suffix}"
        if col1 in df.columns and col2 in df.columns:
            new_a = np.where(swap, df[col2], df[col1])
            new_b = np.where(swap, df[col1], df[col2])
            df[f"a_{suffix}"] = new_a
            df[f"b_{suffix}"] = new_b

    # Assign odds to fighter_a and fighter_b
    # favourite/underdog are named — match to our a/b ordering
    has_odds = df["favourite"].notna()
    df["odds_a"] = np.nan
    df["odds_b"] = np.nan

    fav_is_a = has_odds & (df["favourite"] == df["fighter_a"])
    fav_is_b = has_odds & (df["favourite"] == df["fighter_b"])
    dog_is_a = has_odds & (df["underdog"] == df["fighter_a"])
    dog_is_b = has_odds & (df["underdog"] == df["fighter_b"])

    df.loc[fav_is_a, "odds_a"] = df.loc[fav_is_a, "favourite_odds"]
    df.loc[fav_is_b, "odds_b"] = df.loc[fav_is_b, "favourite_odds"]
    df.loc[dog_is_a, "odds_a"] = df.loc[dog_is_a, "underdog_odds"]
    df.loc[dog_is_b, "odds_b"] = df.loc[dog_is_b, "underdog_odds"]

    # Market probabilities (normalized for overround)
    valid_odds = df["odds_a"].notna() & df["odds_b"].notna()
    df.loc[valid_odds, "implied_a"] = 1.0 / df.loc[valid_odds, "odds_a"]
    df.loc[valid_odds, "implied_b"] = 1.0 / df.loc[valid_odds, "odds_b"]
    total = df["implied_a"] + df["implied_b"]
    df["market_prob"] = df["implied_a"] / total  # probability of fighter_a winning
    df["market_overround"] = total

    # Winner name for reference
    df["winner"] = np.where(df["a_wins"] == 1, df["fighter_a"], df["fighter_b"])

    # Extract method category
    df["method_cat"] = df["method"].str.extract(r"^(KO/TKO|SUB|U-DEC|S-DEC|M-DEC|DQ|Overturned)")
    df["method_cat"] = df["method_cat"].fillna("Other")

    # Context features
    df["is_title_fight"] = df["event_name"].str.contains(
        r"title|championship|interim", case=False, na=False
    ).astype(int)
    df["is_main_event"] = 0  # not available in this dataset

    # Physical diffs
    df["height_diff_cm"] = pd.to_numeric(df["a_height"], errors="coerce") - \
                           pd.to_numeric(df["b_height"], errors="coerce")
    df["reach_diff_cm"] = pd.to_numeric(df["a_reach"], errors="coerce") - \
                          pd.to_numeric(df["b_reach"], errors="coerce")

    # Age at fight time
    for side in ["a", "b"]:
        dob = pd.to_datetime(df[f"{side}_dob"], errors="coerce")
        df[f"age_{side}"] = (df["date"] - dob).dt.days / 365.25

    df["age_diff"] = df["age_a"] - df["age_b"]
    df["age_fighter"] = df[["age_a", "age_b"]].mean(axis=1)  # avg age of bout

    # Stance matchup encoding
    stance_map = {"Orthodox": 0, "Southpaw": 1, "Switch": 2}
    sa = df["a_stance"].map(stance_map).fillna(0)
    sb = df["b_stance"].map(stance_map).fillna(0)
    df["stance_matchup"] = sa * 3 + sb  # 0-8 encoding of matchup

    # Weight class encoding (by weight limit)
    from config import WEIGHT_CLASSES
    df["weight_class_encoded"] = df["weight_class"].map(WEIGHT_CLASSES).fillna(170)

    # Keep useful columns
    keep = [
        "date", "year", "event_name", "weight_class", "weight_class_encoded",
        "fighter_a", "fighter_b", "a_wins", "winner",
        "odds_a", "odds_b", "market_prob", "market_overround",
        "method", "method_cat", "round",
        "height_diff_cm", "reach_diff_cm", "age_diff", "age_fighter", "age_a", "age_b",
        "stance_matchup", "is_title_fight", "is_main_event",
    ]
    # Add per-fighter stats
    for suffix in stat_suffixes:
        for side in ["a", "b"]:
            col = f"{side}_{suffix}"
            if col in df.columns:
                keep.append(col)

    keep = [c for c in keep if c in df.columns]
    result = df[keep].copy().sort_values("date").reset_index(drop=True)

    print(f"  Processed: {len(result)} fights, {result['odds_a'].notna().sum()} with odds")
    print(f"  Date range: {result['date'].min().date()} to {result['date'].max().date()}")
    print(f"  a_wins mean: {result['a_wins'].mean():.3f} (should be ~0.50)")

    return result


def run():
    """Full data ingestion pipeline."""
    print("Loading raw UFC data...")
    raw = download_raw_data()
    print(f"  {len(raw)} raw fights loaded")

    print("Processing...")
    df = process_fights(raw)

    df.to_parquet(FIGHTS_PATH, index=False)
    print(f"Saved to {FIGHTS_PATH}")
    return df


if __name__ == "__main__":
    run()
