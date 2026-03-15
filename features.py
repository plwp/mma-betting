"""Feature engineering for MMA fight prediction.

Builds fighter-level rolling stats, Glicko-2 ratings, physical attributes,
and contextual features from historical fight data.
"""

import math

import numpy as np
import pandas as pd

from config import (
    FEATURE_COLS,
    FEATURE_PATH,
    FIGHTS_PATH,
    GLICKO2_INIT_RATING,
    GLICKO2_INIT_RD,
    GLICKO2_INIT_VOL,
    GLICKO2_TAU,
    GLICKO2_SEASON_RD_INFLATE,
    WEIGHT_CLASSES,
)


def _glicko2_update(rating, rd, vol, opp_rating, opp_rd, score):
    """Single Glicko-2 update. Returns (new_rating, new_rd, new_vol)."""
    mu = (rating - 1500) / 173.7178
    phi = rd / 173.7178
    mu_j = (opp_rating - 1500) / 173.7178
    phi_j = opp_rd / 173.7178

    g_j = 1.0 / math.sqrt(1 + 3 * phi_j**2 / (math.pi**2))
    E_val = 1.0 / (1 + math.exp(-g_j * (mu - mu_j)))

    v = 1.0 / (g_j**2 * E_val * (1 - E_val) + 1e-10)
    delta = v * g_j * (score - E_val)

    a = math.log(vol**2)
    tau = GLICKO2_TAU

    def f(x):
        ex = math.exp(x)
        return (
            ex * (delta**2 - phi**2 - v - ex)
            / (2 * (phi**2 + v + ex)**2)
            - (x - a) / tau**2
        )

    A = a
    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
            if k > 100:
                break
        B = a - k * tau

    f_A, f_B = f(A), f(B)
    for _ in range(50):
        if abs(B - A) < 1e-6:
            break
        C = A + (A - B) * f_A / (f_B - f_A)
        f_C = f(C)
        if f_C * f_B < 0:
            A, f_A = B, f_B
        else:
            f_A /= 2
        B, f_B = C, f_C

    new_vol = math.exp(A / 2)
    phi_star = math.sqrt(phi**2 + new_vol**2)
    new_phi = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    new_mu = mu + new_phi**2 * g_j * (score - E_val)

    return 173.7178 * new_mu + 1500, 173.7178 * new_phi, new_vol


def build_glicko2(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fighter Glicko-2 ratings with inactivity RD inflation."""
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}  # fighter -> (rating, rd, vol, last_fight_date)

    cols = {
        "f1_rating": [], "f1_rd": [],
        "f2_rating": [], "f2_rd": [],
    }

    for row in df.itertuples(index=False):
        f1, f2 = row.fighter_1, row.fighter_2
        date = pd.Timestamp(row.date)

        # Get or init states
        for fighter in (f1, f2):
            if fighter not in ratings:
                ratings[fighter] = (GLICKO2_INIT_RATING, GLICKO2_INIT_RD, GLICKO2_INIT_VOL, date)
            else:
                r, rd, vol, last_date = ratings[fighter]
                # Inflate RD based on inactivity
                days_inactive = (date - last_date).days
                if days_inactive > 0:
                    periods = days_inactive / 90.0
                    rd = min(math.sqrt(rd**2 + (GLICKO2_SEASON_RD_INFLATE * periods)**2),
                             GLICKO2_INIT_RD)
                    ratings[fighter] = (r, rd, vol, last_date)

        r1, rd1, vol1, _ = ratings[f1]
        r2, rd2, vol2, _ = ratings[f2]

        cols["f1_rating"].append(r1)
        cols["f1_rd"].append(rd1)
        cols["f2_rating"].append(r2)
        cols["f2_rd"].append(rd2)

        # Update ratings
        score1 = 1.0 if row.winner == f1 else (0.0 if row.winner == f2 else 0.5)
        new_r1, new_rd1, new_vol1 = _glicko2_update(r1, rd1, vol1, r2, rd2, score1)
        new_r2, new_rd2, new_vol2 = _glicko2_update(r2, rd2, vol2, r1, rd1, 1.0 - score1)

        ratings[f1] = (new_r1, new_rd1, new_vol1, date)
        ratings[f2] = (new_r2, new_rd2, new_vol2, date)

    for col, vals in cols.items():
        df[col] = vals

    df["glicko_rating_diff"] = df["f1_rating"] - df["f2_rating"]
    df["glicko_rd_diff"] = df["f1_rd"] - df["f2_rd"]
    df["glicko_uncertainty"] = np.sqrt(df["f1_rd"]**2 + df["f2_rd"]**2)

    # Glicko-derived win probability
    g_rd = 1.0 / np.sqrt(1 + 3 * (df["f1_rd"]**2 + df["f2_rd"]**2) / (np.pi**2 * 173.7178**2))
    df["glicko_prob"] = 1.0 / (1 + np.exp(-g_rd * (df["f1_rating"] - df["f2_rating"]) / 173.7178))

    return df


def build_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-fighter rolling statistics from fight history.

    TODO: Implement after data format is known. Key stats:
    - sig_strikes_landed_pm, sig_strike_acc, sig_strike_def
    - td_avg, td_acc, td_def
    - sub_avg
    - ko_rate, sub_rate
    - win_pct, recent_form (last 3)
    - days_since_last_fight
    """
    # Placeholder — will implement once we have the actual data schema
    print("  Rolling stats: TODO after data inspection")
    return df


def build_feature_matrix(fights_path: str = FIGHTS_PATH,
                         output_path: str = FEATURE_PATH) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print("Loading fight data...")
    df = pd.read_parquet(fights_path)

    print("Building Glicko-2 ratings...")
    df = build_glicko2(df)

    print("Building rolling stats...")
    df = build_rolling_stats(df)

    # Verify all features present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: missing features (will fill with defaults): {missing}")
        for col in missing:
            df[col] = 0.0

    print(f"Feature matrix: {df.shape}")
    if output_path:
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    build_feature_matrix()
