"""Model training: LogReg + LightGBM + XGBoost + calibrated stacker."""

import os
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLS,
    FEATURE_PATH,
    MODEL_DIR,
    SAMPLE_WEIGHT_HALF_LIFE,
    STACKER_C_VALUES,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)


def _clip_probs(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def _logit(values) -> np.ndarray:
    probs = _clip_probs(values)
    return np.log(probs / (1 - probs))


def temporal_split(df: pd.DataFrame):
    train = df[df["year"] <= TRAIN_END].copy()
    val = df[(df["year"] >= VAL_START) & (df["year"] <= VAL_END)].copy()
    test = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END)].copy()
    return train, val, test


def _compute_sample_weights(years: np.ndarray, max_year: int) -> np.ndarray:
    age = max_year - years.astype(float)
    return np.exp(-np.log(2) * age / SAMPLE_WEIGHT_HALF_LIFE)


def evaluate(y_true, y_prob, label: str, market_prob=None):
    y_prob = _clip_probs(y_prob)
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"\n--- {label} ---")
    print(f"  Log Loss:    {ll:.4f}")
    print(f"  Brier Score: {bs:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    if market_prob is not None:
        market_prob = _clip_probs(market_prob)
        m_ll = log_loss(y_true, market_prob)
        m_bs = brier_score_loss(y_true, market_prob)
        m_acc = accuracy_score(y_true, (market_prob >= 0.5).astype(int))
        print(f"  Market Log Loss:    {m_ll:.4f}")
        print(f"  Market Brier Score: {m_bs:.4f}")
        print(f"  Market Accuracy:    {m_acc:.4f}")
        print(f"  Delta vs Market LL: {m_ll - ll:+.4f}")
    return {"log_loss": ll, "brier_score": bs, "accuracy": acc}


def plot_calibration(y_true, y_prob, label: str, path: str):
    frac_pos, mean_pred = calibration_curve(y_true, _clip_probs(y_prob), n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, "s-", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration: {label}")
    ax.legend()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration plot saved: {path}")


# ---------------------------------------------------------------------------
# EnsemblePredictor
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePredictor:
    """Final prediction bundle for MMA."""

    feature_cols: list[str]
    scaler: StandardScaler
    logreg: LogisticRegression
    lgb_model: lgb.LGBMClassifier
    xgb_model: xgb.XGBClassifier
    stacker: LogisticRegression

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.feature_cols].copy()
        return pd.DataFrame(X, columns=self.feature_cols)

    def _base_probs(self, X) -> dict:
        frame = self._frame(X)
        lr_prob = self.logreg.predict_proba(self.scaler.transform(frame))[:, 1]
        lgb_prob = self.lgb_model.predict_proba(frame)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(frame)[:, 1]
        market_prob = frame["market_prob"].to_numpy(dtype=float)

        return {
            "lr": _clip_probs(lr_prob),
            "lgb": _clip_probs(lgb_prob),
            "xgb": _clip_probs(xgb_prob),
            "market": _clip_probs(market_prob),
        }

    def _stack_features(self, X) -> np.ndarray:
        probs = self._base_probs(X)
        frame = self._frame(X)
        glicko_unc = frame["glicko_uncertainty"].to_numpy(dtype=float)

        return np.column_stack([
            _logit(probs["lr"]),
            _logit(probs["lgb"]),
            _logit(probs["xgb"]),
            _logit(probs["market"]),
            probs["lgb"] - probs["market"],
            probs["xgb"] - probs["market"],
            probs["lr"] - probs["market"],
            glicko_unc,
        ])

    def predict_proba(self, X) -> np.ndarray:
        prob = _clip_probs(
            self.stacker.predict_proba(self._stack_features(X))[:, 1]
        )
        return np.column_stack([1 - prob, prob])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def _tune_logreg(X_train, y_train, X_val, y_val, sample_weight=None):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    candidates = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    best_score = float("inf")
    best_model = None
    best_meta = {}

    for c_value in candidates:
        model = LogisticRegression(max_iter=5000, C=c_value, solver="lbfgs")
        model.fit(X_train_sc, y_train, sample_weight=sample_weight)
        val_prob = model.predict_proba(X_val_sc)[:, 1]
        score = log_loss(y_val, _clip_probs(val_prob))
        if score < best_score:
            best_score = score
            best_model = model
            best_meta = {"C": c_value, "val_log_loss": score}

    return scaler, best_model, best_meta


def _tune_lgbm(X_train, y_train, X_val, y_val, sample_weight=None):
    candidates = [
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
         "num_leaves": 15, "min_child_samples": 40, "subsample": 0.85,
         "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 0.5},
        {"n_estimators": 500, "learning_rate": 0.025, "max_depth": 5,
         "num_leaves": 23, "min_child_samples": 30, "subsample": 0.8,
         "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
         "num_leaves": 7, "min_child_samples": 50, "subsample": 0.9,
         "colsample_bytree": 0.9, "reg_alpha": 0.3, "reg_lambda": 1.5},
    ]
    best_score = float("inf")
    best_model = None
    best_meta = {}

    for params in candidates:
        model = lgb.LGBMClassifier(
            objective="binary", random_state=42, verbose=-1, **params,
        )
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, _clip_probs(val_prob))
        if score < best_score:
            best_score = score
            best_model = model
            best_meta = dict(params)
            best_meta["val_log_loss"] = score
            best_meta["best_iteration_"] = getattr(model, "best_iteration_", None)

    return best_model, best_meta


def _tune_xgboost(X_train, y_train, X_val, y_val, sample_weight=None):
    candidates = [
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
         "min_child_weight": 5, "subsample": 0.85,
         "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 1.0},
        {"n_estimators": 500, "learning_rate": 0.025, "max_depth": 5,
         "min_child_weight": 3, "subsample": 0.8,
         "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.5},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
         "min_child_weight": 8, "subsample": 0.9,
         "colsample_bytree": 0.9, "reg_alpha": 0.3, "reg_lambda": 2.0},
    ]
    best_score = float("inf")
    best_model = None
    best_meta = {}

    for params in candidates:
        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, verbosity=0, early_stopping_rounds=50, **params,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], verbose=False)
        val_prob = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, _clip_probs(val_prob))
        if score < best_score:
            best_score = score
            best_model = model
            best_meta = dict(params)
            best_meta["val_log_loss"] = score
            best_meta["best_iteration"] = getattr(model, "best_iteration", None)

    return best_model, best_meta


# ---------------------------------------------------------------------------
# Stacker
# ---------------------------------------------------------------------------

def _fit_stacker(probs, X_cal, y_cal):
    glicko_unc = X_cal["glicko_uncertainty"].to_numpy(dtype=float)

    stack_X = np.column_stack([
        _logit(probs["lr"]),
        _logit(probs["lgb"]),
        _logit(probs["xgb"]),
        _logit(probs["market"]),
        probs["lgb"] - probs["market"],
        probs["xgb"] - probs["market"],
        probs["lr"] - probs["market"],
        glicko_unc,
    ])

    best_score = float("inf")
    best_model = None
    for c_value in STACKER_C_VALUES:
        model = LogisticRegression(max_iter=5000, C=c_value, solver="lbfgs")
        model.fit(stack_X, y_cal)
        score = log_loss(y_cal, _clip_probs(model.predict_proba(stack_X)[:, 1]))
        if score < best_score:
            best_score = score
            best_model = model

    return best_model


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def fit_model_bundle(train_df, calibration_df):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["a_wins"].to_numpy()
    X_cal = calibration_df[FEATURE_COLS]
    y_cal = calibration_df["a_wins"].to_numpy()

    max_year = int(train_df["year"].max())
    sw = _compute_sample_weights(train_df["year"].to_numpy(), max_year)

    scaler, logreg, logreg_meta = _tune_logreg(X_train, y_train, X_cal, y_cal, sample_weight=sw)
    lgb_model, lgb_meta = _tune_lgbm(X_train, y_train, X_cal, y_cal, sample_weight=sw)
    xgb_model, xgb_meta = _tune_xgboost(X_train, y_train, X_cal, y_cal, sample_weight=sw)

    # Calibration probabilities
    lr_cal_prob = _clip_probs(logreg.predict_proba(scaler.transform(X_cal))[:, 1])
    lgb_cal_prob = _clip_probs(lgb_model.predict_proba(X_cal)[:, 1])
    xgb_cal_prob = _clip_probs(xgb_model.predict_proba(X_cal)[:, 1])
    market_cal_prob = _clip_probs(calibration_df["market_prob"].to_numpy())

    cal_probs = {
        "lr": lr_cal_prob,
        "lgb": lgb_cal_prob,
        "xgb": xgb_cal_prob,
        "market": market_cal_prob,
    }

    stacker = _fit_stacker(cal_probs, X_cal, y_cal)

    predictor = EnsemblePredictor(
        feature_cols=list(FEATURE_COLS),
        scaler=scaler,
        logreg=logreg,
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        stacker=stacker,
    )
    meta = {"logreg": logreg_meta, "lgb": lgb_meta, "xgb": xgb_meta}
    return predictor, meta


# ---------------------------------------------------------------------------
# Main training + evaluation
# ---------------------------------------------------------------------------

def train_models(df=None):
    if df is None:
        df = pd.read_parquet(FEATURE_PATH)

    # Only use fights with odds
    df = df[df["odds_a"].notna() & df["odds_b"].notna()].copy()

    os.makedirs(MODEL_DIR, exist_ok=True)
    train, val, test = temporal_split(df)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    if train.empty or val.empty:
        raise ValueError("Temporal split produced empty train or val")

    predictor, meta = fit_model_bundle(train, val)
    print(f"Selected logistic params: {meta['logreg']}")
    print(f"Selected LightGBM params: {meta['lgb']}")
    print(f"Selected XGBoost params: {meta['xgb']}")

    X_val = val[FEATURE_COLS]
    y_val = val["a_wins"].to_numpy()
    market_val = val["market_prob"].to_numpy()

    val_probs = predictor._base_probs(X_val)
    ens_val_prob = predictor.predict_proba(X_val)[:, 1]

    evaluate(y_val, market_val, "Market (Val)")
    evaluate(y_val, val_probs["lr"], "LogReg (Val)", market_val)
    evaluate(y_val, val_probs["lgb"], "LightGBM (Val)", market_val)
    evaluate(y_val, val_probs["xgb"], "XGBoost (Val)", market_val)
    evaluate(y_val, ens_val_prob, "Ensemble (Val)", market_val)

    if not test.empty:
        X_test = test[FEATURE_COLS]
        y_test = test["a_wins"].to_numpy()
        market_test = test["market_prob"].to_numpy()

        test_probs = predictor._base_probs(X_test)
        ens_test_prob = predictor.predict_proba(X_test)[:, 1]

        evaluate(y_test, market_test, "Market (Test)")
        evaluate(y_test, test_probs["lr"], "LogReg (Test)", market_test)
        evaluate(y_test, test_probs["lgb"], "LightGBM (Test)", market_test)
        evaluate(y_test, test_probs["xgb"], "XGBoost (Test)", market_test)
        evaluate(y_test, ens_test_prob, "Ensemble (Test)", market_test)

        plot_calibration(y_test, market_test, "Market",
                         os.path.join(MODEL_DIR, "calibration_market.png"))
        plot_calibration(y_test, ens_test_prob, "Ensemble",
                         os.path.join(MODEL_DIR, "calibration_ensemble.png"))

    importance = pd.Series(
        predictor.lgb_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nFeature Importance (LightGBM):")
    print(importance.to_string())

    joblib.dump(predictor, os.path.join(MODEL_DIR, "model_bundle.pkl"))
    print(f"\nModel bundle saved to {MODEL_DIR}/")

    return {"predictor": predictor, "meta": meta}


if __name__ == "__main__":
    train_models()
