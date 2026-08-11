from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .costs import execution_cost_breakdown

LONG = 1
SHORT = -1


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(
        np.asarray(probability, dtype=float),
        1e-5,
        1 - 1e-5,
    )
    return np.log(values / (1 - values)).reshape(-1, 1)


def _effective_sample_weight(
    X: pd.DataFrame,
    sample_weight: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(sample_weight, dtype=float)
    if "model_sample_weight_multiplier" not in X:
        return weights
    multiplier = pd.to_numeric(
        X["model_sample_weight_multiplier"],
        errors="coerce",
    ).fillna(1.0).clip(0.01, 1.0).to_numpy(dtype=float)
    return weights * multiplier


@dataclass
class BlendClassifier:
    tree_classifier: Any
    linear_classifier: Any
    tree_weight: float
    logistic_weight: float
    calibrator: Any | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "BlendClassifier":
        y = np.asarray(y, dtype=int)
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(np.unique(y)) != 2:
            raise ValueError("A binary classifier requires both target classes")
        split = int(len(X) * 0.82)
        can_calibrate = (
            split >= 200
            and len(X) - split >= 100
            and len(np.unique(y[:split])) == 2
            and len(np.unique(y[split:])) == 2
        )
        if can_calibrate:
            self._fit_base(
                X.iloc[:split],
                y[:split],
                sample_weight[:split],
            )
            raw = self._raw_probability(X.iloc[split:])
            calibrator = LogisticRegression(
                C=0.75,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
            calibrator.fit(
                _logit(raw),
                y[split:],
                sample_weight=sample_weight[split:],
            )
            self.calibrator = calibrator
        else:
            self.calibrator = None
            self._fit_base(X, y, sample_weight)
        return self

    def _fit_base(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> None:
        self.tree_classifier.fit(
            X,
            y,
            sample_weight=sample_weight,
        )
        self.linear_classifier.fit(
            X,
            y,
            clf__sample_weight=sample_weight,
        )

    def _raw_probability(self, X: pd.DataFrame) -> np.ndarray:
        tree_probability = self.tree_classifier.predict_proba(X)[:, 1]
        linear_probability = self.linear_classifier.predict_proba(X)[:, 1]
        return np.clip(
            self.tree_weight * tree_probability
            + self.logistic_weight * linear_probability,
            1e-5,
            1 - 1e-5,
        )

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._raw_probability(X)
        if self.calibrator is None:
            return raw
        return np.clip(
            self.calibrator.predict_proba(_logit(raw))[:, 1],
            1e-4,
            1 - 1e-4,
        )


@dataclass
class DirectionalEventHead:
    direction: int
    feature_columns: list[str]
    success_classifier: BlendClassifier
    tradeability_classifier: BlendClassifier
    return_regressor: Any
    minimum_samples: int
    available: bool = False
    fitted_samples: int = 0

    def fit(
        self,
        X: pd.DataFrame,
        y_success: np.ndarray,
        y_tradeable: np.ndarray,
        y_return: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "DirectionalEventHead":
        valid = (
            np.isfinite(y_success)
            & np.isfinite(y_tradeable)
            & np.isfinite(y_return)
        )
        indices = np.flatnonzero(valid)
        self.fitted_samples = int(len(indices))
        if (
            len(indices) < self.minimum_samples
            or len(np.unique(y_success[indices].astype(int))) != 2
            or len(np.unique(y_tradeable[indices].astype(int))) != 2
        ):
            self.available = False
            return self
        event_X = X.iloc[indices].reindex(columns=self.feature_columns)
        event_weight = _effective_sample_weight(
            event_X,
            sample_weight[indices],
        )
        self.success_classifier.fit(
            event_X,
            y_success[indices].astype(int),
            event_weight,
        )
        self.tradeability_classifier.fit(
            event_X,
            y_tradeable[indices].astype(int),
            event_weight,
        )
        self.return_regressor.fit(
            event_X,
            y_return[indices],
            sample_weight=event_weight,
        )
        self.available = True
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.available:
            return {
                "p_success": np.full(len(X), 0.5, dtype=float),
                "p_tradeable": np.full(len(X), 0.5, dtype=float),
                "event_return": np.zeros(len(X), dtype=float),
            }
        event_X = X.reindex(columns=self.feature_columns)
        return {
            "p_success": self.success_classifier.predict_probability(event_X),
            "p_tradeable": (
                self.tradeability_classifier.predict_probability(event_X)
            ),
            "event_return": self.return_regressor.predict(event_X),
        }


@dataclass
class HorizonModel:
    horizon: int
    general_feature_columns: list[str]
    general_direction_classifier: BlendClassifier
    general_return_regressor: Any
    long_head: DirectionalEventHead
    short_head: DirectionalEventHead
    general_available: bool = False

    def fit_general(
        self,
        X: pd.DataFrame,
        y_direction: np.ndarray,
        y_return: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "HorizonModel":
        general_X = X.reindex(columns=self.general_feature_columns)
        effective_weight = _effective_sample_weight(
            general_X,
            sample_weight,
        )
        self.general_direction_classifier.fit(
            general_X,
            y_direction,
            effective_weight,
        )
        self.general_return_regressor.fit(
            general_X,
            y_return,
            sample_weight=effective_weight,
        )
        self.general_available = True
        return self

    def fit_direction(
        self,
        direction: int,
        X: pd.DataFrame,
        y_success: np.ndarray,
        y_tradeable: np.ndarray,
        y_return: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "HorizonModel":
        head = self.long_head if direction == LONG else self.short_head
        head.fit(
            X,
            y_success,
            y_tradeable,
            y_return,
            sample_weight,
        )
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.general_available:
            general_X = X.reindex(columns=self.general_feature_columns)
            probability_up = (
                self.general_direction_classifier.predict_probability(general_X)
            )
            general_return = self.general_return_regressor.predict(general_X)
        else:
            probability_up = np.full(len(X), 0.5, dtype=float)
            general_return = np.zeros(len(X), dtype=float)
        long_output = self.long_head.predict(X)
        short_output = self.short_head.predict(X)
        direction = pd.to_numeric(
            X.get("event_direction", pd.Series(0, index=X.index)),
            errors="coerce",
        ).fillna(0).to_numpy(dtype=int)
        continuation = np.where(
            direction == LONG,
            long_output["p_success"],
            np.where(
                direction == SHORT,
                short_output["p_success"],
                0.5,
            ),
        )
        tradeability = np.where(
            direction == LONG,
            long_output["p_tradeable"],
            np.where(
                direction == SHORT,
                short_output["p_tradeable"],
                0.5,
            ),
        )
        event_return = np.where(
            direction == LONG,
            long_output["event_return"],
            np.where(
                direction == SHORT,
                short_output["event_return"],
                0.0,
            ),
        )
        return {
            "p_up": probability_up,
            "general_return": general_return,
            "p_continuation": continuation,
            "p_tradeable": tradeability,
            "event_return": event_return,
            "long_p_continuation": long_output["p_success"],
            "long_p_tradeable": long_output["p_tradeable"],
            "long_event_return": long_output["event_return"],
            "short_p_continuation": short_output["p_success"],
            "short_p_tradeable": short_output["p_tradeable"],
            "short_event_return": short_output["event_return"],
        }


@dataclass
class HourlyModelBundle:
    model_id: str
    created_at: str
    provider: str
    symbol: str
    feature_columns: list[str]
    event_feature_columns: list[str]
    horizons: list[int]
    trade_horizons: list[int]
    horizon_weights: dict[int, float]
    models: dict[int, HorizonModel]
    metrics: dict[str, Any]
    qualification: dict[str, Any]
    training_range: dict[str, str]
    event_inventory: dict[str, Any]
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 5

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        X = frame.reindex(columns=self.feature_columns)
        probabilities: dict[int, float] = {}
        general_returns: dict[int, float] = {}
        continuation: dict[int, float] = {}
        tradeability: dict[int, float] = {}
        event_returns: dict[int, float] = {}
        long_outputs: dict[int, dict[str, float]] = {}
        short_outputs: dict[int, dict[str, float]] = {}
        for horizon in self.horizons:
            output = self.models[horizon].predict(X)
            probabilities[horizon] = float(output["p_up"][-1])
            general_returns[horizon] = float(
                output["general_return"][-1]
            )
            continuation[horizon] = float(
                output["p_continuation"][-1]
            )
            tradeability[horizon] = float(
                output["p_tradeable"][-1]
            )
            event_returns[horizon] = float(
                output["event_return"][-1]
            )
            long_outputs[horizon] = {
                "p_success": float(output["long_p_continuation"][-1]),
                "p_tradeable": float(output["long_p_tradeable"][-1]),
                "event_return": float(output["long_event_return"][-1]),
            }
            short_outputs[horizon] = {
                "p_success": float(output["short_p_continuation"][-1]),
                "p_tradeable": float(output["short_p_tradeable"][-1]),
                "event_return": float(output["short_event_return"][-1]),
            }

        latest = frame.iloc[-1]
        event_direction = int(latest.get("event_direction", 0))
        direction_name = (
            "LONG"
            if event_direction == LONG
            else "SHORT"
            if event_direction == SHORT
            else "NONE"
        )
        qualified_by_direction = self.qualification.get(
            "qualified_directions",
            {},
        )
        qualified = [
            int(horizon)
            for horizon in qualified_by_direction.get(direction_name, [])
        ]
        trade_candidates = [
            horizon
            for horizon in self.trade_horizons
            if horizon in self.horizons
        ]
        candidates = qualified or trade_candidates or list(self.horizons)
        strategy_cfg = self.config_snapshot.get("strategy", {})
        stress_cost = execution_cost_breakdown(strategy_cfg)[
            "stress_cost_bps"
        ] / 10_000

        horizon_scores: dict[int, float] = {}
        for horizon in self.horizons:
            direction_metrics = (
                self.metrics.get(str(horizon), {})
                .get("directions", {})
                .get(direction_name, {})
            )
            fold_consistency = float(
                direction_metrics.get("positive_fold_fraction", 0.0)
            )
            quality = 0.20 + 0.80 * fold_consistency
            predicted_net = event_returns[horizon] - stress_cost
            horizon_scores[horizon] = (
                continuation[horizon]
                * tradeability[horizon]
                * max(predicted_net, 0.0)
                * quality
            )

        if any(horizon_scores.get(horizon, 0.0) > 0 for horizon in candidates):
            selected_horizon = max(
                candidates,
                key=lambda horizon: horizon_scores.get(horizon, 0.0),
            )
        else:
            selected_horizon = max(
                candidates,
                key=lambda horizon: (
                    continuation.get(horizon, 0.5)
                    * tradeability.get(horizon, 0.5)
                ),
            )

        direction_score = sum(
            self.horizon_weights.get(horizon, 0.0)
            * (2 * probabilities[horizon] - 1)
            for horizon in self.horizons
        )
        direction = "UP" if direction_score >= 0 else "DOWN"
        confidence = float(0.5 + abs(direction_score) / 2)
        votes = [
            "UP" if probabilities[horizon] >= 0.5 else "DOWN"
            for horizon in self.horizons
            if self.horizon_weights.get(horizon, 0.0) > 0
        ]
        agreement = (
            max(votes.count("UP"), votes.count("DOWN")) / len(votes)
            if votes
            else 0.5
        )
        event_votes = [
            continuation[horizon] >= 0.5
            for horizon in self.trade_horizons
            if horizon in continuation
        ]
        event_agreement = float(np.mean(event_votes)) if event_votes else 0.0
        trade_direction = (
            "UP"
            if event_direction == LONG
            else "DOWN"
            if event_direction == SHORT
            else direction
        )
        selected_event_return = float(event_returns[selected_horizon])
        signed_event_returns = {
            horizon: (
                float(event_returns[horizon] * event_direction)
                if event_direction
                else 0.0
            )
            for horizon in self.horizons
        }
        return {
            "direction": direction,
            "confidence": confidence,
            "agreement": float(agreement),
            "event_agreement": event_agreement,
            "trade_direction": trade_direction,
            "direction_qualified": selected_horizon in qualified,
            "qualified_trade_horizons": qualified,
            "probabilities": probabilities,
            "returns": general_returns,
            "continuation": continuation,
            "tradeability": tradeability,
            "event_returns": event_returns,
            "absolute_event_returns": signed_event_returns,
            "long_outputs": long_outputs,
            "short_outputs": short_outputs,
            "horizon_scores": horizon_scores,
            "selected_horizon": int(selected_horizon),
            "expected_return": signed_event_returns[selected_horizon],
            "expected_event_aligned_return": selected_event_return,
            "event_id": latest.get("event_id"),
            "event_type": str(latest.get("event_type", "NONE")),
            "event_direction": event_direction,
            "event_direction_name": direction_name,
            "regime": str(latest.get("regime", "UNKNOWN")),
            "breakout_level": latest.get("breakout_level"),
            "breakout_source": latest.get("breakout_source"),
            "triangle_type": latest.get("triangle_type"),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_bundle(path: Path) -> HourlyModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, HourlyModelBundle):
        raise TypeError(
            f"Unexpected model artifact: {type(bundle)!r}"
        )
    compatible = (
        str(getattr(bundle, "model_id", "")).startswith(
            "directional-breakout-hourly-"
        )
        and all(
            hasattr(model, "long_head") and hasattr(model, "short_head")
            for model in getattr(bundle, "models", {}).values()
        )
    )
    if int(getattr(bundle, "schema_version", 1)) < 5 or not compatible:
        raise RuntimeError(
            "The model artifact predates deterministic directional "
            "breakout schema v5. Run a full retraining cycle."
        )
    return bundle


def latest_bundle(settings: Settings) -> HourlyModelBundle:
    path = settings.path("model_dir") / "latest.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "No deterministic directional breakout model found. "
            "Run: btc-regime train"
        )
    return load_bundle(path)


def build_horizon_model(
    settings: Settings,
    horizon: int,
    general_feature_columns: list[str],
    event_feature_columns: list[str],
) -> HorizonModel:
    model_cfg = settings.section("model")
    long_cfg = settings.section("long_model")
    short_cfg = settings.section("short_model")
    return HorizonModel(
        horizon=horizon,
        general_feature_columns=general_feature_columns,
        general_direction_classifier=_build_classifier(model_cfg),
        general_return_regressor=_build_regressor(model_cfg),
        long_head=DirectionalEventHead(
            direction=LONG,
            feature_columns=event_feature_columns,
            success_classifier=_build_classifier(long_cfg),
            tradeability_classifier=_build_classifier(long_cfg),
            return_regressor=_build_regressor(long_cfg),
            minimum_samples=int(
                long_cfg.get("minimum_fit_samples", 300)
            ),
        ),
        short_head=DirectionalEventHead(
            direction=SHORT,
            feature_columns=event_feature_columns,
            success_classifier=_build_classifier(short_cfg),
            tradeability_classifier=_build_classifier(short_cfg),
            return_regressor=_build_regressor(short_cfg),
            minimum_samples=int(
                short_cfg.get("minimum_fit_samples", 300)
            ),
        ),
    )


def _build_classifier(cfg: dict[str, Any]) -> BlendClassifier:
    tree_weight = float(cfg.get("tree_weight", 0.70))
    logistic_weight = float(cfg.get("logistic_weight", 0.30))
    total = tree_weight + logistic_weight
    tree = HistGradientBoostingClassifier(
        learning_rate=float(cfg.get("learning_rate", 0.03)),
        max_iter=int(cfg.get("max_iter", 260)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 11)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 28)),
        l2_regularization=float(cfg.get("l2_regularization", 3.0)),
        class_weight="balanced",
        early_stopping=False,
    )
    linear = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(cfg.get("logistic_c", 0.15)),
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return BlendClassifier(
        tree_classifier=tree,
        linear_classifier=linear,
        tree_weight=tree_weight / total,
        logistic_weight=logistic_weight / total,
    )


def _build_regressor(cfg: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=float(cfg.get("learning_rate", 0.03)),
        max_iter=int(cfg.get("max_iter", 260)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 11)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 28)),
        l2_regularization=float(cfg.get("l2_regularization", 3.0)),
        early_stopping=False,
    )
