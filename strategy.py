"""Betting strategy for MMA: edge threshold on both fighters."""

import pandas as pd

from config import EDGE_THRESHOLD, MAX_ODDS, MIN_MODEL_PROB
from sizing import edge, kelly_stake


class BettingStrategy:
    """Bet either fighter when model edge exceeds threshold."""

    def __init__(
        self,
        edge_threshold: float = EDGE_THRESHOLD,
        max_odds: float = MAX_ODDS,
        min_model_prob: float = MIN_MODEL_PROB,
    ):
        self.edge_threshold = edge_threshold
        self.max_odds = max_odds
        self.min_model_prob = min_model_prob

    def _check_side(self, row, side, model_prob, odds_key, won, bankroll):
        odds = row.get(odds_key)
        if odds is None or pd.isna(odds):
            return None
        if model_prob < self.min_model_prob or odds > self.max_odds:
            return None

        e = edge(model_prob, odds)
        if e <= self.edge_threshold:
            return None

        stake = kelly_stake(model_prob, odds, bankroll)
        if stake <= 0:
            return None

        return {
            "side": side,
            "model_prob": model_prob,
            "odds": odds,
            "edge": e,
            "stake": stake,
            "won": won,
        }

    def select_bets(self, row, prob_a: float, bankroll: float) -> list:
        candidates = []

        bet_a = self._check_side(
            row, "fighter_a", prob_a, "odds_a",
            row["a_wins"] == 1, bankroll,
        )
        if bet_a:
            candidates.append(bet_a)

        bet_b = self._check_side(
            row, "fighter_b", 1.0 - prob_a, "odds_b",
            row["a_wins"] == 0, bankroll,
        )
        if bet_b:
            candidates.append(bet_b)

        if not candidates:
            return []
        # Take the single best edge bet per fight
        return [max(candidates, key=lambda b: b["edge"])]
