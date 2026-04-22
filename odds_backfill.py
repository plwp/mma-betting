"""Backfill missing historical odds from shortlikeafox/ultimate_ufc_dataset.

The original jansen88/ufc-data feed (via betmma.tips) stopped updating in
September 2023, leaving ~2.5 years of fights (2024-2026) with no odds in the
training set. `model.py` filters those rows out during training, which meant
the model was effectively frozen on pre-September-2023 data.

This module downloads the actively-maintained ultimate_ufc_dataset CSV
(Red/Blue corner format, American odds) and fills in missing `odds_a`,
`odds_b`, `market_prob`, `market_overround` on `fights.parquet`. Existing
odds are never overwritten, only gaps are filled.

Run after `scrape_results.py` to fold in any new events before rebuilding
the feature matrix.
"""

import io
import os
import time

import numpy as np
import pandas as pd
import requests

from config import DATA_DIR, FIGHTS_PATH, ULTIMATE_UFC_URL


ODDS_SRC_CACHE = os.path.join(DATA_DIR, "ultimate_ufc_master.csv")
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours — dataset updates weekly


def _american_to_decimal(american) -> float:
    """Convert American odds (-130, +102) to decimal odds."""
    if pd.isna(american):
        return np.nan
    val = float(american)
    if val == 0:
        return np.nan
    if val > 0:
        return val / 100.0 + 1.0
    return 100.0 / abs(val) + 1.0


def _download_csv() -> pd.DataFrame:
    """Fetch the ultimate_ufc_dataset CSV, cached to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    fresh = os.path.exists(ODDS_SRC_CACHE) and (
        time.time() - os.path.getmtime(ODDS_SRC_CACHE) < CACHE_TTL_SECONDS
    )
    if not fresh:
        print(f"  Downloading {ULTIMATE_UFC_URL}...")
        resp = requests.get(ULTIMATE_UFC_URL, timeout=60)
        resp.raise_for_status()
        with open(ODDS_SRC_CACHE, "wb") as f:
            f.write(resp.content)
    return pd.read_csv(ODDS_SRC_CACHE)


def _pair_key(a, b) -> tuple:
    """Order-independent fighter-pair key."""
    return tuple(sorted([str(a).strip(), str(b).strip()]))


def backfill_odds(fights: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `fights` with missing odds filled from the upstream CSV.

    Matching is by (date, fighter-pair). Fighter ordering in upstream uses
    Red/Blue corner; we map R_odds/B_odds to our a/b ordering by name.
    """
    src = _download_csv()
    src = src[["date", "R_fighter", "B_fighter", "R_odds", "B_odds"]].copy()
    src["date"] = pd.to_datetime(src["date"], errors="coerce").dt.normalize()
    src = src.dropna(subset=["date", "R_fighter", "B_fighter"])
    src["R_dec"] = src["R_odds"].map(_american_to_decimal)
    src["B_dec"] = src["B_odds"].map(_american_to_decimal)
    src = src.dropna(subset=["R_dec", "B_dec"])
    src["pair"] = src.apply(
        lambda r: _pair_key(r["R_fighter"], r["B_fighter"]), axis=1
    )

    df = fights.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["pair"] = df.apply(
        lambda r: _pair_key(r["fighter_a"], r["fighter_b"]), axis=1
    )

    before = int(df["odds_a"].notna().sum())
    target = df["odds_a"].isna() | df["odds_b"].isna()

    # Build a dict keyed on (date, pair) -> (R_fighter, R_dec, B_dec) to avoid
    # MultiIndex .loc surprises when a tuple of strings is treated as columns.
    lookup = {
        (d, p): (str(r).strip(), rd, bd)
        for d, p, r, rd, bd in zip(
            src["date"], src["pair"], src["R_fighter"], src["R_dec"], src["B_dec"]
        )
    }

    filled = 0
    for idx in df[target].index:
        key = (df.at[idx, "date"], df.at[idx, "pair"])
        entry = lookup.get(key)
        if entry is None:
            continue
        r_fighter, r_dec, b_dec = entry
        if str(df.at[idx, "fighter_a"]).strip() == r_fighter:
            df.at[idx, "odds_a"] = r_dec
            df.at[idx, "odds_b"] = b_dec
        else:
            df.at[idx, "odds_a"] = b_dec
            df.at[idx, "odds_b"] = r_dec
        filled += 1

    # Recompute market_prob / overround on every row that now has both sides.
    has_both = df["odds_a"].notna() & df["odds_b"].notna()
    implied_a = 1.0 / df.loc[has_both, "odds_a"]
    implied_b = 1.0 / df.loc[has_both, "odds_b"]
    total = implied_a + implied_b
    df.loc[has_both, "market_prob"] = implied_a / total
    df.loc[has_both, "market_overround"] = total

    df = df.drop(columns=["pair"])

    after = int(df["odds_a"].notna().sum())
    print(
        f"  Odds coverage: {before} -> {after} (+{after - before})"
        f"  [backfilled {filled} rows from upstream]"
    )
    return df


def run():
    print("Loading fights...")
    fights = pd.read_parquet(FIGHTS_PATH)
    print(f"  {len(fights)} fights, {fights['odds_a'].notna().sum()} with odds")

    print("Backfilling odds from shortlikeafox/ultimate_ufc_dataset...")
    fights = backfill_odds(fights)

    # Year-by-year coverage report
    yr = (
        fights.assign(has=fights["odds_a"].notna())
        .groupby("year")["has"].agg(["sum", "count"])
    )
    yr["pct"] = (yr["sum"] / yr["count"] * 100).round(1)
    print("\nOdds coverage by year:")
    print(yr.tail(15).to_string())

    fights.to_parquet(FIGHTS_PATH, index=False)
    print(f"\nSaved to {FIGHTS_PATH}")


if __name__ == "__main__":
    run()
