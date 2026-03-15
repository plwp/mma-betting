"""Walk-forward backtesting engine for MMA."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    EDGE_THRESHOLD, FEATURE_COLS, INITIAL_BANKROLL,
    MAX_ODDS, MIN_MODEL_PROB, MIN_STAKE,
)
from model import fit_model_bundle, _clip_probs
from strategy import BettingStrategy


def walk_forward_backtest(
    df: pd.DataFrame,
    start_year: int = 2017,
    end_year: int = 2023,
    initial_bankroll: float = INITIAL_BANKROLL,
    edge_threshold: float = EDGE_THRESHOLD,
    max_odds: float = MAX_ODDS,
    min_model_prob: float = MIN_MODEL_PROB,
) -> dict:
    """Walk-forward backtest with yearly retraining."""
    # Only use fights with odds
    df = df[df["odds_a"].notna() & df["odds_b"].notna()].copy()

    bankroll = float(initial_bankroll)
    bet_log = []
    bankroll_history = [(df[df["year"] == start_year]["date"].min(), bankroll)]

    strategy = BettingStrategy(
        edge_threshold=edge_threshold,
        max_odds=max_odds,
        min_model_prob=min_model_prob,
    )

    for year in range(start_year, end_year + 1):
        train_data = df[df["year"] <= year - 3].copy()
        cal_data = df[(df["year"] >= year - 2) & (df["year"] <= year - 1)].copy()
        test_data = df[df["year"] == year].copy().sort_values("date", kind="mergesort")

        if len(train_data) < 100 or len(cal_data) < 40 or len(test_data) == 0:
            print(f"  Skipping {year}: insufficient data "
                  f"(train={len(train_data)}, cal={len(cal_data)}, test={len(test_data)})")
            continue

        predictor, meta = fit_model_bundle(train_data, cal_data)
        probs = predictor.predict_proba(test_data[FEATURE_COLS])[:, 1]

        print(f"  {year}: {len(train_data)} train / {len(cal_data)} cal / {len(test_data)} test")

        year_bets = 0
        year_pnl = 0.0

        # Daily bankroll lock
        current_date = None
        daily_start_bankroll = bankroll
        pending_pnl = 0.0

        for idx, row in test_data.reset_index(drop=True).iterrows():
            if bankroll < MIN_STAKE:
                print(f"  Stopping in {year}: bankroll ${bankroll:.2f} below minimum")
                break

            if current_date is not None and row["date"] != current_date:
                bankroll += pending_pnl
                daily_start_bankroll = bankroll
                pending_pnl = 0.0
            current_date = row["date"]

            candidates = strategy.select_bets(row, float(probs[idx]), daily_start_bankroll)
            if not candidates:
                continue

            for candidate in candidates:
                pnl = candidate["stake"] * (candidate["odds"] - 1) if candidate["won"] else -candidate["stake"]
                pending_pnl += pnl
                year_pnl += pnl
                year_bets += 1

                bet_log.append({
                    "date": row["date"],
                    "year": year,
                    "fighter_a": row["fighter_a"],
                    "fighter_b": row["fighter_b"],
                    "side": candidate["side"],
                    "model_prob": candidate["model_prob"],
                    "market_prob": row["market_prob"],
                    "odds": candidate["odds"],
                    "edge": candidate["edge"],
                    "stake": candidate["stake"],
                    "won": candidate["won"],
                    "pnl": pnl,
                    "bankroll": bankroll + pending_pnl,
                })
                bankroll_history.append((row["date"], bankroll + pending_pnl))

        bankroll += pending_pnl
        print(f"  {year}: {year_bets} bets, P&L ${year_pnl:+.2f}, Bankroll ${bankroll:.2f}")
        if bankroll < MIN_STAKE:
            break

    bets_df = pd.DataFrame(bet_log)
    return _compute_summary(bets_df, bankroll_history, initial_bankroll)


def _compute_summary(bets_df, bankroll_history, initial_bankroll):
    if bets_df.empty:
        print("No bets placed.")
        return {"bets_df": bets_df, "bankroll_history": bankroll_history}

    total_staked = float(bets_df["stake"].sum())
    total_pnl = float(bets_df["pnl"].sum())
    n_bets = len(bets_df)
    n_wins = int(bets_df["won"].sum())
    win_rate = n_wins / n_bets

    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    bankroll_return = (bankroll_history[-1][1] / initial_bankroll) - 1

    bankroll_series = pd.Series([b for _, b in bankroll_history], dtype=float)
    peak = bankroll_series.cummax()
    drawdown = (bankroll_series - peak) / peak.replace(0, np.nan)
    max_dd = float(drawdown.min())

    daily_pnl = bets_df.groupby("date")["pnl"].sum()
    sharpe = 0.0
    if len(daily_pnl) > 1 and daily_pnl.std(ddof=0) > 0:
        sharpe = float((daily_pnl.mean() / daily_pnl.std(ddof=0)) * np.sqrt(len(daily_pnl)))

    summary = {
        "total_bets": n_bets, "wins": n_wins, "win_rate": win_rate,
        "total_staked": total_staked, "total_pnl": total_pnl,
        "roi": roi, "bankroll_return": bankroll_return,
        "max_drawdown": max_dd, "sharpe": sharpe,
        "final_bankroll": bankroll_history[-1][1],
    }

    print("\n=== Backtest Results ===")
    print(f"  Total Bets:       {n_bets}")
    print(f"  Win Rate:         {win_rate:.1%}")
    print(f"  Total Staked:     ${total_staked:,.2f}")
    print(f"  Total P&L:        ${total_pnl:+,.2f}")
    print(f"  ROI on Stakes:    {roi:+.1%}")
    print(f"  Bankroll Return:  {bankroll_return:+.1%}")
    print(f"  Max Drawdown:     {max_dd:.1%}")
    print(f"  Sharpe-like:      {sharpe:.2f}")
    print(f"  Final Bankroll:   ${bankroll_history[-1][1]:,.2f}")

    return {"summary": summary, "bets_df": bets_df, "bankroll_history": bankroll_history}


def plot_bankroll(bankroll_history, path="models/bankroll_curve.png"):
    dates = [d for d, _ in bankroll_history]
    values = [v for _, v in bankroll_history]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, linewidth=1.5)
    ax.axhline(y=values[0], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("MMA Backtest Bankroll Curve")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Bankroll curve saved: {path}")


if __name__ == "__main__":
    df = pd.read_parquet("data/feature_matrix.parquet")
    results = walk_forward_backtest(df)
    if results.get("bankroll_history"):
        plot_bankroll(results["bankroll_history"])
