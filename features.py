"""Feature engineering for MMA fight prediction.

Builds fighter-level Glicko-2 ratings, Elo, rolling career stats,
and contextual features from historical fight data.
"""

import math
import os

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
)


def _glicko2_update(rating, rd, vol, opp_rating, opp_rd, score):
    """Single Glicko-2 update."""
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


def _elo_expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def build_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fighter Glicko-2 and Elo ratings."""
    df = df.sort_values("date").reset_index(drop=True)

    # State: fighter -> (glicko_rating, glicko_rd, glicko_vol, elo, last_date)
    state = {}

    def _get(fighter, date):
        if fighter not in state:
            state[fighter] = (GLICKO2_INIT_RATING, GLICKO2_INIT_RD,
                              GLICKO2_INIT_VOL, 1500.0, date)
            return state[fighter]
        r, rd, vol, elo, last = state[fighter]
        # Inflate RD for inactivity
        days = (date - last).days
        if days > 0:
            periods = days / 90.0
            rd = min(math.sqrt(rd**2 + (GLICKO2_SEASON_RD_INFLATE * periods)**2),
                     GLICKO2_INIT_RD)
            state[fighter] = (r, rd, vol, elo, last)
        return state[fighter]

    cols = {k: [] for k in [
        "glicko_a", "glicko_b", "rd_a", "rd_b", "elo_a", "elo_b",
    ]}

    for row in df.itertuples(index=False):
        date = pd.Timestamp(row.date)
        fa, fb = row.fighter_a, row.fighter_b

        r_a, rd_a, vol_a, elo_a, _ = _get(fa, date)
        r_b, rd_b, vol_b, elo_b, _ = _get(fb, date)

        cols["glicko_a"].append(r_a)
        cols["glicko_b"].append(r_b)
        cols["rd_a"].append(rd_a)
        cols["rd_b"].append(rd_b)
        cols["elo_a"].append(elo_a)
        cols["elo_b"].append(elo_b)

        score_a = float(row.a_wins)

        # Glicko update
        nr_a, nrd_a, nvol_a = _glicko2_update(r_a, rd_a, vol_a, r_b, rd_b, score_a)
        nr_b, nrd_b, nvol_b = _glicko2_update(r_b, rd_b, vol_b, r_a, rd_a, 1.0 - score_a)

        # Elo update (K=32)
        exp_a = _elo_expected(elo_a, elo_b)
        new_elo_a = elo_a + 32 * (score_a - exp_a)
        new_elo_b = elo_b + 32 * ((1 - score_a) - (1 - exp_a))

        state[fa] = (nr_a, nrd_a, nvol_a, new_elo_a, date)
        state[fb] = (nr_b, nrd_b, nvol_b, new_elo_b, date)

    for col, vals in cols.items():
        df[col] = vals

    df["glicko_rating_diff"] = df["glicko_a"] - df["glicko_b"]
    df["glicko_rd_diff"] = df["rd_a"] - df["rd_b"]
    df["glicko_uncertainty"] = np.sqrt(df["rd_a"]**2 + df["rd_b"]**2)
    df["elo_diff"] = df["elo_a"] - df["elo_b"]

    # Glicko-derived win probability
    g = 1.0 / np.sqrt(1 + 3 * (df["rd_a"]**2 + df["rd_b"]**2) / (np.pi**2 * 173.7178**2))
    df["glicko_prob"] = 1.0 / (1 + np.exp(-g * (df["glicko_a"] - df["glicko_b"]) / 173.7178))

    return df


def build_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-fighter rolling career statistics."""
    df = df.sort_values("date").reset_index(drop=True)

    # Track per-fighter history
    fighter_history = {}  # fighter -> list of fight dicts

    stat_cols = {
        "win_pct": [], "win_streak": [],
        "ko_rate": [], "sub_rate": [],
        "recent_form_3": [],
        "days_since_last_fight": [],
    }
    # Duplicate for both sides
    result_cols = {f"{k}_{side}": [] for k in stat_cols for side in ["a", "b"]}

    for row in df.itertuples(index=False):
        date = pd.Timestamp(row.date)

        for side in ["a", "b"]:
            fighter = row.fighter_a if side == "a" else row.fighter_b
            won = (row.a_wins == 1) if side == "a" else (row.a_wins == 0)

            hist = fighter_history.get(fighter, [])

            if len(hist) == 0:
                result_cols[f"win_pct_{side}"].append(0.5)
                result_cols[f"win_streak_{side}"].append(0)
                result_cols[f"ko_rate_{side}"].append(0.0)
                result_cols[f"sub_rate_{side}"].append(0.0)
                result_cols[f"recent_form_3_{side}"].append(0.5)
                result_cols[f"days_since_last_fight_{side}"].append(365.0)
            else:
                wins = [h["won"] for h in hist]
                result_cols[f"win_pct_{side}"].append(np.mean(wins))

                # Current win/loss streak (computed from history only, no leakage)
                streak = 0
                if wins:
                    last_outcome = wins[-1]
                    for w in reversed(wins):
                        if w == last_outcome:
                            streak += 1
                        else:
                            break
                    if not last_outcome:
                        streak = -streak
                result_cols[f"win_streak_{side}"].append(streak)

                methods = [h["method_cat"] for h in hist if h["won"]]
                n_wins = len(methods)
                result_cols[f"ko_rate_{side}"].append(
                    sum(1 for m in methods if m == "KO/TKO") / max(n_wins, 1)
                )
                result_cols[f"sub_rate_{side}"].append(
                    sum(1 for m in methods if m == "SUB") / max(n_wins, 1)
                )

                recent = wins[-3:] if len(wins) >= 3 else wins
                result_cols[f"recent_form_3_{side}"].append(np.mean(recent))

                last_date = hist[-1]["date"]
                result_cols[f"days_since_last_fight_{side}"].append(
                    max((date - last_date).days, 0)
                )

            # Record this fight in history (after computing features)
            fighter_history.setdefault(fighter, []).append({
                "date": date,
                "won": won,
                "method_cat": row.method_cat,
            })

    for col, vals in result_cols.items():
        df[col] = vals

    # Compute diffs
    for stat in stat_cols:
        df[f"{stat}_diff"] = df[f"{stat}_a"] - df[f"{stat}_b"]

    return df


FIGHT_STATS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "fight_stats.parquet"
)


def build_rolling_fight_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fighter rolling averages from scraped per-fight stats.

    Uses data/fight_stats.parquet (one row per fighter per fight) to compute
    cumulative averages of sig strikes, takedowns, etc. Only uses fights
    BEFORE the current fight (no future leakage).
    """
    if not os.path.exists(FIGHT_STATS_PATH):
        print("  WARNING: fight_stats.parquet not found, skipping rolling fight stats")
        for col in ["sig_str_pm_diff", "sig_str_acc_diff", "sig_str_def_diff",
                     "td_pm_diff", "td_acc_diff", "sub_att_pm_diff"]:
            df[col] = 0.0
        return df

    stats = pd.read_parquet(FIGHT_STATS_PATH)
    stats = stats.sort_values("event_date").reset_index(drop=True)

    # Build cumulative stats per fighter
    # Track: cumulative sig_str_landed, sig_str_attempted, td_landed, td_attempted,
    #         sub_att, total fights, total time (approx)
    fighter_cum = {}  # fighter -> {stat: cumulative_value, n_fights: int}

    # Index stats by (fighter, event_date, event_name) for lookup
    fighter_stats_at = {}  # fighter -> list of (date, stats_dict)
    for row in stats.itertuples(index=False):
        fighter_stats_at.setdefault(row.fighter, []).append({
            "date": row.event_date,
            "sig_str_landed": row.sig_str_landed,
            "sig_str_attempted": row.sig_str_attempted,
            "total_str_landed": row.total_str_landed,
            "total_str_attempted": row.total_str_attempted,
            "td_landed": row.td_landed,
            "td_attempted": row.td_attempted,
            "sub_att": row.sub_att,
            "knockdowns": row.knockdowns,
            "ctrl_seconds": row.ctrl_seconds,
        })

    # Process each fight in chronological order
    # For each fighter, compute rolling averages from their PAST fights only
    fighter_processed = {}  # fighter -> number of fights processed

    def _get_rolling(fighter, fight_date):
        """Get rolling averages for fighter using only fights before fight_date."""
        history = fighter_stats_at.get(fighter, [])
        past = [h for h in history if h["date"] < fight_date]

        n = len(past)
        if n == 0:
            return {
                "sig_str_pm": 0.0, "sig_str_acc": 0.0,
                "td_pm": 0.0, "td_acc": 0.0,
                "sub_att_pm": 0.0, "kd_pm": 0.0,
            }

        total_sig_landed = sum(h["sig_str_landed"] for h in past)
        total_sig_attempted = sum(h["sig_str_attempted"] for h in past)
        total_str_landed = sum(h["total_str_landed"] for h in past)
        total_td_landed = sum(h["td_landed"] for h in past)
        total_td_attempted = sum(h["td_attempted"] for h in past)
        total_sub = sum(h["sub_att"] for h in past)
        total_kd = sum(h["knockdowns"] for h in past)

        return {
            "sig_str_pm": total_sig_landed / n,  # per fight avg
            "sig_str_acc": total_sig_landed / max(total_sig_attempted, 1),
            "td_pm": total_td_landed / n,
            "td_acc": total_td_landed / max(total_td_attempted, 1),
            "sub_att_pm": total_sub / n,
            "kd_pm": total_kd / n,
        }

    result_cols = {
        "sig_str_pm_a": [], "sig_str_pm_b": [],
        "sig_str_acc_a": [], "sig_str_acc_b": [],
        "td_pm_a": [], "td_pm_b": [],
        "td_acc_a": [], "td_acc_b": [],
        "sub_att_pm_a": [], "sub_att_pm_b": [],
        "kd_pm_a": [], "kd_pm_b": [],
    }

    for row in df.itertuples(index=False):
        date = pd.Timestamp(row.date)
        for side in ["a", "b"]:
            fighter = row.fighter_a if side == "a" else row.fighter_b
            rolling = _get_rolling(fighter, date)
            for stat_key, val in rolling.items():
                result_cols[f"{stat_key}_{side}"].append(val)

    for col, vals in result_cols.items():
        df[col] = vals

    # Compute diffs
    for stat in ["sig_str_pm", "sig_str_acc", "td_pm", "td_acc", "sub_att_pm", "kd_pm"]:
        df[f"{stat}_diff"] = df[f"{stat}_a"] - df[f"{stat}_b"]

    return df


def build_feature_matrix(fights_path: str = FIGHTS_PATH,
                         output_path: str = FEATURE_PATH) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print("Loading fight data...")
    df = pd.read_parquet(fights_path)

    print("Building Glicko-2 and Elo ratings...")
    df = build_ratings(df)

    print("Building rolling fighter stats...")
    df = build_rolling_stats(df)

    print("Building rolling fight stats from scraped data...")
    df = build_rolling_fight_stats(df)

    # Fill NaN with defaults
    fill_values = {
        "market_prob": 0.5,
        "market_overround": 1.0,
        "height_diff_cm": 0.0,
        "reach_diff_cm": 0.0,
        "age_diff": 0.0,
        "age_fighter": 30.0,
        "win_pct_diff": 0.0,
        "win_streak_diff": 0,
        "ko_rate_diff": 0.0,
        "sub_rate_diff": 0.0,
        "recent_form_3_diff": 0.0,
        "days_since_last_fight_diff": 0.0,
        "sig_str_pm_diff": 0.0,
        "sig_str_acc_diff": 0.0,
        "td_pm_diff": 0.0,
        "td_acc_diff": 0.0,
        "sub_att_pm_diff": 0.0,
        "kd_pm_diff": 0.0,
        "stance_matchup": 0,
        "weight_class_encoded": 170,
        "is_title_fight": 0,
        "is_main_event": 0,
        "glicko_rating_diff": 0.0,
        "glicko_rd_diff": 0.0,
        "glicko_uncertainty": 495.0,
        "elo_diff": 0.0,
    }
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Verify features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Only keep fights with odds for backtesting
    has_odds = df["odds_a"].notna() & df["odds_b"].notna()
    print(f"  {has_odds.sum()} fights with odds (of {len(df)} total)")
    print(f"  Feature matrix shape: {df.shape}")

    if output_path:
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    build_feature_matrix()
