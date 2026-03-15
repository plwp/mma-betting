"""Kelly criterion bet sizing."""

from config import KELLY_FRACTION, MAX_BET_FRACTION, MIN_STAKE


def kelly_stake(prob: float, odds: float, bankroll: float,
                fraction: float = KELLY_FRACTION,
                max_frac: float = MAX_BET_FRACTION,
                min_stake: float = MIN_STAKE) -> float:
    """Compute Kelly criterion stake."""
    b = odds - 1
    if b <= 0:
        return 0.0

    q = 1 - prob
    kelly_frac = (b * prob - q) / b

    if kelly_frac <= 0:
        return 0.0

    stake_frac = min(kelly_frac * fraction, max_frac)
    stake = stake_frac * bankroll

    if stake < min_stake:
        return 0.0

    return round(stake, 2)


def edge(prob: float, odds: float) -> float:
    """Compute edge: model_prob * odds - 1 (EV per dollar)."""
    return prob * odds - 1
