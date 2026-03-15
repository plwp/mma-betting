# MMA Betting Model

Ensemble ML system for UFC fight prediction and value bet identification, with Pinnacle steam detection for AU book latency arbitrage.

## Thesis

1. **Model edge**: Combine Glicko-2 ratings, fighter stats, physical attributes, and market odds via a calibrated stacker (same architecture as [afl-betting](https://github.com/plwp/afl-betting)). UFC markets are less efficient than major team sports due to name recognition bias, lower liquidity, and higher variance.

2. **Steam edge**: Pinnacle is the sharpest UFC book. When their line moves, AU books (Sportsbet, TAB, Ladbrokes) lag 30-120 minutes behind. Monitor Pinnacle in real-time, bet AU books before they adjust.

## Data Sources

| Source | Data | Status |
|--------|------|--------|
| ufcstats.com | Fight results, round-by-round stats | Via `ufcscraper` |
| BestFightOdds | Historical odds from 12+ bookmakers | Via `ufcscraper` |
| The Odds API | Live odds for steam detection | API key required |
| Kaggle (jerzyszocik) | Daily UFC odds dataset | Backup source |

## Architecture

```
data_ingest.py     Scrape fights + odds, merge
features.py        Glicko-2, rolling fighter stats, physical/contextual features
model.py           Ensemble (LogReg + LightGBM + XGBoost + MarginReg + stacker)
backtest.py        Walk-forward backtesting
strategy.py        Bet selection with edge thresholds
sizing.py          Kelly criterion
steam_monitor.py   Pinnacle line movement -> AU book latency arb
config.py          All parameters and feature columns
```

## Quick Start

```bash
pip install -r requirements.txt

# Scrape data
python data_ingest.py

# Build features + backtest
python features.py
python run_backtest.py

# Live steam monitor (requires ODDS_API_KEY in .env)
python steam_monitor.py --poll-interval 300
```

## Key Differences from AFL Model

| Aspect | AFL | MMA |
|--------|-----|-----|
| Outcome | Binary (home/away win) | Binary (fighter 1/2 win) |
| Matches/year | ~200 | ~500 |
| Rating system | Elo + Glicko-2 (team) | Glicko-2 (individual, with inactivity RD inflation) |
| Variance | Lower (team sport) | Higher (one-punch KOs) |
| Market efficiency | High | Moderate (name recognition bias, casual money) |
| Key features | Market odds, Elo, form, venue | Market odds, Glicko, physical stats, style matchup |
| Steam opportunity | Minimal | Yes (Pinnacle -> AU book lag) |

## Status

Scaffolded. Next steps:
1. Run `data_ingest.py` to scrape initial dataset
2. Inspect data format and wire up merge logic
3. Build rolling fighter stats in `features.py`
4. Port ensemble + backtest from AFL project

## License

GPL-3.0
