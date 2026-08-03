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


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(
        np.asarray(probability, dtype=float),
        1e-5,
        1 - 1e-5,
    )
    return np.log(values / (1 - values)).reshape(-1, 1)


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
        split = int(len(X) * 0.80)
        can_calibrate = (
            split >= 100
            and len(X) - split >= 60
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
                C=1.0,
                max_iter=500,
                class_weight="balanced",
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
class HorizonModel:
    horizon: int
    general_direction_classifier: BlendClassifier
    general_return_regressor: Any
    event_continuation_classifier: BlendClassifier
    event_tradeability_classifier: BlendClassifier
    event_return_regressor: Any
    minimum_event_samples: int
    event_models_available: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y_direction: np.ndarray,
        y_return: np.ndarray,
        sample_weight: np.ndarray,
        event_mask: np.ndarray,
        y_continuation: np.ndarray,
        y_tradeable: np.ndarray,
        y_event_return: np.ndarray,
    ) -> "HorizonModel":
        self.general_direction_classifier.fit(
            X,
            y_direction,
            sample_weight,
        )
        self.general_return_regressor.fit(
            X,
            y_return,
            sample_weight=sample_weight,
        )
        valid_event = (
            event_mask
            & np.isfinite(y_continuation)
            & np.isfinite(y_tradeable)
            & np.isfinite(y_event_return)
        )
        event_indices = np.flatnonzero(valid_event)
        if (
            len(event_indices) >= self.minimum_event_samples
            and len(
                np.unique(
                    y_continuation[event_indices].astype(int)
                )
            )
            == 2
            and len(
                np.unique(y_tradeable[event_indices].astype(int))
            )
            == 2
        ):
            event_X = X.iloc[event_indices]
            event_weight = sample_weight[event_indices]
            self.event_continuation_classifier.fit(
                event_X,
                y_continuation[event_indices].astype(int),
                event_weight,
            )
            self.event_tradeability_classifier.fit(
                event_X,
                y_tradeable[event_indices].astype(int),
                event_weight,
            )
            self.event_return_regressor.fit(
                event_X,
                y_event_return[event_indices],
                sample_weight=event_weight,
            )
            self.event_models_available = True
        else:
            self.event_models_available = False
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        probability_up = (
            self.general_direction_classifier.predict_probability(X)
        )
        general_return = self.general_return_regressor.predict(X)
        if self.event_models_available:
            continuation = (
                self.event_continuation_classifier.predict_probability(X)
            )
            tradeability = (
                self.event_tradeability_classifier.predict_probability(X)
            )
            event_return = self.event_return_regressor.predict(X)
        else:
            continuation = np.full(len(X), 0.5, dtype=float)
            tradeability = np.full(len(X), 0.5, dtype=float)
            event_return = np.zeros(len(X), dtype=float)
        return {
            "p_up": probability_up,
            "general_return": general_return,
            "p_continuation": continuation,
            "p_tradeable": tradeability,
            "event_return": event_return,
        }


@dataclass
class HourlyModelBundle:
    model_id: str
    created_at: str
    provider: str
    symbol: str
    feature_columns: list[str]
    horizons: list[int]
    horizon_weights: dict[int, float]
    models: dict[int, HorizonModel]
    metrics: dict[str, Any]
    qualification: dict[str, Any]
    training_range: dict[str, str]
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 4

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        X = frame.reindex(columns=self.feature_columns)
        probabilities: dict[int, float] = {}
        general_returns: dict[int, float] = {}
        continuation: dict[int, float] = {}
        tradeability: dict[int, float] = {}
        event_returns: dict[int, float] = {}
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

        latest = frame.iloc[-1]
        event_direction = int(latest.get("event_direction", 0))
        qualified = [
            int(horizon)
            for horizon in self.qualification.get(
                "qualified_horizons",
                [],
            )
        ]
        candidates = qualified or list(self.horizons)
        strategy_cfg = self.config_snapshot.get("strategy", {})
        stress_cost = execution_cost_breakdown(strategy_cfg)[
            "stress_cost_bps"
        ] / 10_000

        horizon_scores: dict[int, float] = {}
        for horizon in self.horizons:
            fold_consistency = float(
                self.metrics.get(str(horizon), {}).get(
                    "positive_fold_fraction",
                    0.0,
                )
            )
            quality = 0.25 + 0.75 * fold_consistency
            predicted_net = event_returns[horizon] - stress_cost
            horizon_scores[horizon] = (
                continuation[horizon]
                * tradeability[horizon]
                * max(predicted_net, 0.0)
                * quality
            )

        if any(horizon_scores[horizon] > 0 for horizon in candidates):
            selected_horizon = max(
                candidates,
                key=lambda horizon: horizon_scores[horizon],
            )
        else:
            selected_horizon = max(
                candidates,
                key=lambda horizon: (
                    continuation[horizon]
                    * tradeability[horizon]
                ),
            )

        direction_score = sum(
            self.horizon_weights[horizon]
            * (2 * probabilities[horizon] - 1)
            for horizon in self.horizons
        )
        direction = "UP" if direction_score >= 0 else "DOWN"
        confidence = float(0.5 + abs(direction_score) / 2)
        votes = [
            "UP" if probabilities[horizon] >= 0.5 else "DOWN"
            for horizon in self.horizons
        ]
        agreement = max(
            votes.count("UP"),
            votes.count("DOWN"),
        ) / len(votes)
        event_agreement = float(
            np.mean(
                [
                    continuation[horizon] >= 0.5
                    for horizon in self.horizons
                ]
            )
        )
        trade_direction = (
            "UP"
            if event_direction > 0
            else "DOWN"
            if event_direction < 0
            else direction
        )
        selected_event_return = float(
            event_returns[selected_horizon]
        )
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
            "probabilities": probabilities,
            "returns": general_returns,
            "continuation": continuation,
            "tradeability": tradeability,
            "event_returns": event_returns,
            "absolute_event_returns": signed_event_returns,
            "horizon_scores": horizon_scores,
            "selected_horizon": int(selected_horizon),
            "expected_return": signed_event_returns[
                selected_horizon
            ],
            "expected_event_aligned_return": selected_event_return,
            "event_id": latest.get("event_id"),
            "event_type": str(
                latest.get("event_type", "NONE")
            ),
            "event_direction": event_direction,
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
            "structure-breakout-hourly-"
        )
        and all(
            hasattr(model, "event_continuation_classifier")
            for model in getattr(bundle, "models", {}).values()
        )
    )
    if int(getattr(bundle, "schema_version", 1)) < 4 or not compatible:
        raise RuntimeError(
            "The model artifact predates structural breakout v4. "
            "Run a full retraining cycle."
        )
    return bundle


def latest_bundle(settings: Settings) -> HourlyModelBundle:
    path = settings.path("model_dir") / "latest.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "No structural breakout model found. Run: btc-regime train"
        )
    return load_bundle(path)


def build_horizon_model(
    settings: Settings,
    horizon: int,
) -> HorizonModel:
    cfg = settings.section("model")
    random_state = int(cfg.get("random_state", 42)) + int(horizon)
    tree_weight = float(cfg.get("tree_weight", 0.65))
    logistic_weight = float(cfg.get("logistic_weight", 0.35))
    total = tree_weight + logistic_weight

    def blend(seed_offset: int) -> BlendClassifier:
        tree = HistGradientBoostingClassifier(
            learning_rate=float(cfg.get("learning_rate", 0.035)),
            max_iter=int(cfg.get("max_iter", 320)),
            max_leaf_nodes=int(cfg.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(
                cfg.get("min_samples_leaf", 36)
            ),
            l2_regularization=float(
                cfg.get("l2_regularization", 2.0)
            ),
            class_weight="balanced",
            random_state=random_state + seed_offset,
            early_stopping=True,
            validation_fraction=0.12,
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
                        C=0.20,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state + seed_offset,
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

    def regressor(
        seed_offset: int,
        min_leaf: int,
    ) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=float(cfg.get("learning_rate", 0.035)),
            max_iter=int(cfg.get("max_iter", 320)),
            max_leaf_nodes=int(cfg.get("max_leaf_nodes", 15)),
            min_samples_leaf=min_leaf,
            l2_regularization=float(
                cfg.get("l2_regularization", 2.0)
            ),
            random_state=random_state + seed_offset,
            early_stopping=True,
            validation_fraction=0.12,
        )

    min_leaf = int(cfg.get("min_samples_leaf", 36))
    return HorizonModel(
        horizon=horizon,
        general_direction_classifier=blend(0),
        general_return_regressor=regressor(10, min_leaf),
        event_continuation_classifier=blend(100),
        event_tradeability_classifier=blend(200),
        event_return_regressor=regressor(
            300,
            max(14, min_leaf // 2),
        ),
        minimum_event_samples=int(
            cfg.get("minimum_training_events", 120)
        ),
    )
