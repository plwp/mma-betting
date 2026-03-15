"""Generate predictions for upcoming UFC fights using live odds."""

import os

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from config import EDGE_THRESHOLD, FEATURE_COLS, MAX_ODDS, MIN_MODEL_PROB, MODEL_DIR
from model import EnsemblePredictor  # noqa: F401 — needed for pickle
from sizing import edge, kelly_stake

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BANKROLL = 1000.0  # Set your current bankroll


def fetch_upcoming_odds() -> list:
    """Fetch upcoming MMA odds from The Odds API."""
    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "au,us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"API requests remaining: {remaining}")
    return resp.json()


def _best_odds(event: dict) -> dict:
    """Extract best available odds per fighter across all bookmakers."""
    best = {}
    books = {}
    for bm in event.get("bookmakers", []):
        market = next((m for m in bm["markets"] if m["key"] == "h2h"), None)
        if not market:
            continue
        for outcome in market["outcomes"]:
            name = outcome["name"]
            price = outcome["price"]
            if name not in best or price > best[name]:
                best[name] = price
                books[name] = bm["key"]
    return best, books


def load_predictor():
    """Load the trained model bundle."""
    path = os.path.join(MODEL_DIR, "model_bundle.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Run model.py first.")
    return joblib.load(path)


def build_fighter_features(fighter_a: str, fighter_b: str, odds_a: float,
                           odds_b: float, predictor) -> pd.DataFrame:
    """Build feature row for a fight using historical data."""
    # Load historical fight data to compute ratings
    fights = pd.read_parquet("data/feature_matrix.parquet")

    # Find most recent Glicko/Elo ratings for each fighter
    state = {}
    for _, row in fights.sort_values("date").iterrows():
        for side, fighter in [("a", row["fighter_a"]), ("b", row["fighter_b"])]:
            state[fighter] = {
                "glicko": row[f"glicko_{side}"],
                "rd": row[f"rd_{side}"],
                "elo": row[f"elo_{side}"],
                "win_pct": row.get(f"win_pct_{side}", 0.5),
                "win_streak": row.get(f"win_streak_{side}", 0),
                "ko_rate": row.get(f"ko_rate_{side}", 0.0),
                "sub_rate": row.get(f"sub_rate_{side}", 0.0),
                "recent_form_3": row.get(f"recent_form_3_{side}", 0.5),
                "days_since_last_fight": row.get(f"days_since_last_fight_{side}", 365),
                "height": row.get(f"a_height" if side == "a" else "b_height", None),
                "reach": row.get(f"a_reach" if side == "a" else "b_reach", None),
                "age": row.get(f"age_{side}", 30),
                "sig_str_pm": row.get(f"sig_str_pm_{side}", 0.0),
                "sig_str_acc": row.get(f"sig_str_acc_{side}", 0.0),
                "td_pm": row.get(f"td_pm_{side}", 0.0),
                "td_acc": row.get(f"td_acc_{side}", 0.0),
                "sub_att_pm": row.get(f"sub_att_pm_{side}", 0.0),
                "kd_pm": row.get(f"kd_pm_{side}", 0.0),
            }

    sa = state.get(fighter_a, {})
    sb = state.get(fighter_b, {})

    if not sa or not sb:
        return None

    # Market probabilities
    implied_a = 1.0 / odds_a
    implied_b = 1.0 / odds_b
    total = implied_a + implied_b
    market_prob = implied_a / total

    glicko_a = sa.get("glicko", 1500)
    glicko_b = sb.get("glicko", 1500)
    rd_a = sa.get("rd", 350)
    rd_b = sb.get("rd", 350)

    row = {
        "glicko_rating_diff": glicko_a - glicko_b,
        "glicko_rd_diff": rd_a - rd_b,
        "glicko_uncertainty": np.sqrt(rd_a**2 + rd_b**2),
        "elo_diff": sa.get("elo", 1500) - sb.get("elo", 1500),
        "market_prob": market_prob,
        "market_overround": total,
        "height_diff_cm": float(sa.get("height", 0) or 0) - float(sb.get("height", 0) or 0),
        "reach_diff_cm": float(sa.get("reach", 0) or 0) - float(sb.get("reach", 0) or 0),
        "age_diff": sa.get("age", 30) - sb.get("age", 30),
        "age_fighter": (sa.get("age", 30) + sb.get("age", 30)) / 2,
        "win_pct_diff": sa.get("win_pct", 0.5) - sb.get("win_pct", 0.5),
        "win_streak_diff": sa.get("win_streak", 0) - sb.get("win_streak", 0),
        "ko_rate_diff": sa.get("ko_rate", 0.0) - sb.get("ko_rate", 0.0),
        "sub_rate_diff": sa.get("sub_rate", 0.0) - sb.get("sub_rate", 0.0),
        "sig_str_pm_diff": sa.get("sig_str_pm", 0.0) - sb.get("sig_str_pm", 0.0),
        "sig_str_acc_diff": sa.get("sig_str_acc", 0.0) - sb.get("sig_str_acc", 0.0),
        "td_pm_diff": sa.get("td_pm", 0.0) - sb.get("td_pm", 0.0),
        "td_acc_diff": sa.get("td_acc", 0.0) - sb.get("td_acc", 0.0),
        "sub_att_pm_diff": sa.get("sub_att_pm", 0.0) - sb.get("sub_att_pm", 0.0),
        "kd_pm_diff": sa.get("kd_pm", 0.0) - sb.get("kd_pm", 0.0),
        "days_since_last_fight_diff": sa.get("days_since_last_fight", 365) - sb.get("days_since_last_fight", 365),
        "recent_form_3_diff": sa.get("recent_form_3", 0.5) - sb.get("recent_form_3", 0.5),
        "is_title_fight": 0,
        "is_main_event": 0,
        "weight_class_encoded": 170,
        "stance_matchup": 0,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


def predict_upcoming():
    """Score all upcoming fights and find value bets."""
    print("Loading model...")
    predictor = load_predictor()

    print("Fetching live odds...")
    events = fetch_upcoming_odds()
    print(f"  {len(events)} events found\n")

    # Load historical data for fighter lookup
    fights = pd.read_parquet("data/feature_matrix.parquet")

    # Build fighter state from historical data
    state = {}
    for _, row in fights.sort_values("date").iterrows():
        for side in ["a", "b"]:
            fighter = row[f"fighter_{side}"]
            state[fighter] = {
                "glicko": row.get(f"glicko_{side}", 1500),
                "rd": row.get(f"rd_{side}", 350),
                "elo": row.get(f"elo_{side}", 1500),
                "win_pct": row.get(f"win_pct_{side}", 0.5),
                "win_streak": row.get(f"win_streak_{side}", 0),
                "ko_rate": row.get(f"ko_rate_{side}", 0.0),
                "sub_rate": row.get(f"sub_rate_{side}", 0.0),
                "recent_form_3": row.get(f"recent_form_3_{side}", 0.5),
                "days_since_last_fight": row.get(f"days_since_last_fight_{side}", 365),
                "sig_str_pm": row.get(f"sig_str_pm_{side}", 0.0),
                "sig_str_acc": row.get(f"sig_str_acc_{side}", 0.0),
                "td_pm": row.get(f"td_pm_{side}", 0.0),
                "td_acc": row.get(f"td_acc_{side}", 0.0),
                "sub_att_pm": row.get(f"sub_att_pm_{side}", 0.0),
                "kd_pm": row.get(f"kd_pm_{side}", 0.0),
            }

    known_fighters = set(state.keys())
    bets = []
    skipped = 0

    for event in events:
        home = event["home_team"]
        away = event["away_team"]
        commence = event["commence_time"][:10]

        best_odds, best_books = _best_odds(event)
        if home not in best_odds or away not in best_odds:
            skipped += 1
            continue

        odds_a = best_odds[home]
        odds_b = best_odds[away]

        # Match fighter names to our database
        fa = home if home in known_fighters else None
        fb = away if away in known_fighters else None

        if not fa or not fb:
            skipped += 1
            continue

        sa, sb = state[fa], state[fb]
        implied_a = 1.0 / odds_a
        implied_b = 1.0 / odds_b
        total_implied = implied_a + implied_b
        market_prob = implied_a / total_implied

        row = {
            "glicko_rating_diff": sa["glicko"] - sb["glicko"],
            "glicko_rd_diff": sa["rd"] - sb["rd"],
            "glicko_uncertainty": np.sqrt(sa["rd"]**2 + sb["rd"]**2),
            "elo_diff": sa["elo"] - sb["elo"],
            "market_prob": market_prob,
            "market_overround": total_implied,
            "height_diff_cm": 0.0,
            "reach_diff_cm": 0.0,
            "age_diff": 0.0,
            "age_fighter": 30.0,
            "win_pct_diff": sa["win_pct"] - sb["win_pct"],
            "win_streak_diff": sa["win_streak"] - sb["win_streak"],
            "ko_rate_diff": sa["ko_rate"] - sb["ko_rate"],
            "sub_rate_diff": sa["sub_rate"] - sb["sub_rate"],
            "sig_str_pm_diff": sa["sig_str_pm"] - sb["sig_str_pm"],
            "sig_str_acc_diff": sa["sig_str_acc"] - sb["sig_str_acc"],
            "td_pm_diff": sa["td_pm"] - sb["td_pm"],
            "td_acc_diff": sa["td_acc"] - sb["td_acc"],
            "sub_att_pm_diff": sa["sub_att_pm"] - sb["sub_att_pm"],
            "kd_pm_diff": sa["kd_pm"] - sb["kd_pm"],
            "days_since_last_fight_diff": sa["days_since_last_fight"] - sb["days_since_last_fight"],
            "recent_form_3_diff": sa["recent_form_3"] - sb["recent_form_3"],
            "is_title_fight": 0,
            "is_main_event": 0,
            "weight_class_encoded": 170,
            "stance_matchup": 0,
        }

        X = pd.DataFrame([row])[FEATURE_COLS]
        prob_a = float(predictor.predict_proba(X)[0, 1])
        prob_b = 1.0 - prob_a

        edge_a = edge(prob_a, odds_a)
        edge_b = edge(prob_b, odds_b)

        has_bet = False
        for fighter, prob, odds_val, e, book in [
            (fa, prob_a, odds_a, edge_a, best_books.get(home, "")),
            (fb, prob_b, odds_b, edge_b, best_books.get(away, "")),
        ]:
            if e > EDGE_THRESHOLD and prob >= MIN_MODEL_PROB and odds_val <= MAX_ODDS:
                stake = kelly_stake(prob, odds_val, BANKROLL)
                if stake > 0:
                    bets.append({
                        "date": commence,
                        "fight": f"{fa} vs {fb}",
                        "bet_on": fighter,
                        "model_prob": prob,
                        "market_prob": implied_a / total_implied if fighter == fa else implied_b / total_implied,
                        "odds": odds_val,
                        "book": book,
                        "edge": e,
                        "kelly_stake": stake,
                    })
                    has_bet = True

        if not has_bet:
            # Still print the fight info
            print(f"  {fa} vs {fb} ({commence})")
            print(f"    Model: {prob_a:.1%} / {prob_b:.1%}  |  "
                  f"Market: {market_prob:.1%} / {1-market_prob:.1%}  |  "
                  f"Odds: {odds_a:.2f} / {odds_b:.2f}  |  "
                  f"Edge: {edge_a:+.1%} / {edge_b:+.1%}")

    print(f"\n{'='*70}")
    print(f"Scanned {len(events)} events, {skipped} skipped (unknown fighters)")

    if bets:
        print(f"\n*** VALUE BETS FOUND ({len(bets)}) ***\n")
        bets_df = pd.DataFrame(bets).sort_values("edge", ascending=False)
        for _, b in bets_df.iterrows():
            print(f"  {b['fight']}")
            print(f"    BET: {b['bet_on']} @ {b['odds']:.2f} ({b['book']})")
            print(f"    Model: {b['model_prob']:.1%} vs Market: {b['market_prob']:.1%}")
            print(f"    Edge: {b['edge']:+.1%}  |  Kelly stake: ${b['kelly_stake']:.2f}")
            print()
    else:
        print("\nNo value bets found at current odds.")

    return bets


if __name__ == "__main__":
    predict_upcoming()
