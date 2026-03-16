"""Configuration constants for MMA betting system."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FIGHTS_PATH = os.path.join(DATA_DIR, "fights.parquet")
ODDS_PATH = os.path.join(DATA_DIR, "odds.parquet")
FEATURE_PATH = os.path.join(DATA_DIR, "feature_matrix.parquet")

# --- Data Sources ---
# ufcstats.com: official UFC stats (round-by-round)
UFC_STATS_BASE = "http://ufcstats.com/statistics/events/completed?page=all"
# BestFightOdds: historical closing odds from 12+ bookmakers
BFO_BASE = "https://www.bestfightodds.com"

# --- Glicko-2 Parameters ---
GLICKO2_INIT_RATING = 1500.0
GLICKO2_INIT_RD = 350.0
GLICKO2_INIT_VOL = 0.06
GLICKO2_TAU = 0.5
GLICKO2_SEASON_RD_INFLATE = 30.0  # RD inflation per 90 days of inactivity

# --- Temporal Splits ---
TRAIN_END = 2021
VAL_START = 2022
VAL_END = 2023
TEST_START = 2024
TEST_END = 2025

# --- Betting Parameters ---
KELLY_FRACTION = 0.25
MAX_BET_FRACTION = 0.05
MIN_STAKE = 5.0
EDGE_THRESHOLD = 0.10
MAX_ODDS = 4.0  # MMA has more upsets than AFL, wider range
MIN_MODEL_PROB = 0.52  # lower than AFL — MMA is less predictable
INITIAL_BANKROLL = 1000.0
SAMPLE_WEIGHT_HALF_LIFE = 2.0  # years; MMA evolves faster than AFL
STACKER_C_VALUES = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]

# --- Feature Columns ---
FEATURE_COLS = [
    # Ratings
    "glicko_rating_diff",
    "glicko_rd_diff",
    "glicko_uncertainty",
    "elo_diff",
    # Market
    "market_prob",
    "market_overround",
    # Physical
    "height_diff_cm",
    "reach_diff_cm",
    "age_diff",
    # Rolling stats (computed from fight history, no future leakage)
    "win_streak_diff",
    "ko_rate_diff",
    "sub_rate_diff",
    # Per-fight stats (scraped from ufcstats, cumulative rolling averages)
    "sig_str_pm_diff",
    "sig_str_acc_diff",
    "td_pm_diff",
    "td_acc_diff",
    "sub_att_pm_diff",
    "kd_pm_diff",
    # Form
    "days_since_last_fight_diff",
    # Context
    "weight_class_encoded",
    # Stance matchup
    "stance_matchup",
]

# --- Odds API (for live scanning) ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# --- Pinnacle Steam Detection ---
PINNACLE_POLL_INTERVAL = 300  # seconds between Pinnacle odds checks
STEAM_THRESHOLD = 0.03  # 3% implied prob move = steam

# --- Weight Classes ---
WEIGHT_CLASSES = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Women's Strawweight": 115,
    "Women's Flyweight": 125,
    "Women's Bantamweight": 135,
    "Women's Featherweight": 145,
}
