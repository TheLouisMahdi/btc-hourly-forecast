from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor

from .config import Settings
from .model import HourlyModelBundle

PRICE_ADAPTIVE_SCHEMA_VERSION = 2
PRICE_VECTOR_LIMIT = 8.0


def _direction_estimator(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=2e-4,
        l1_ratio=0.05,
        learning_rate="optimal",
        average=True,
        random_state=seed,
    )


def _return_estimator(seed: int) -> SGDRegressor:
    return SGDRegressor(
        loss="huber",
        penalty="elasticnet",
        alpha=2e-4,
        l1_ratio=0.05,
        learning_rate="invscaling",
        eta0=0.003,
        power_t=0.25,
        average=True,
        random_state=seed,
    )


@dataclass
class PriceAdaptiveState:
    schema_version: int
    champion_model_id: str
    created_at: str
    updated_at: str
    direction_estimator: SGDClassifier
    return_estimator: SGDRegressor
    initialized: bool = False
    samples_seen: int = 0
    last_trained_open_time: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    rebase_count: int = 0


class PriceAdaptiveEngine:
    """Shadow online learner with strict evidence-based blend gates.

    The online model is allowed to influence a forecast only when it is
    measurably better than the current batch champion over the locked rolling
    evaluation window. Being merely "not much worse" is no longer sufficient.
    """

    def __init__(self, settings: Settings, bundle: HourlyModelBundle) -> None:
        self.settings = settings
        self.bundle = bundle
        self.config = settings.section("forecast")
        self.path = settings.path("price_adaptive_state")
        self.state = self._load_or_create()
        if self.state.champion_model_id != bundle.model_id:
            self.state = self._new_state(self.state.rebase_count + 1)
            self.save()

    def _load_or_create(self) -> PriceAdaptiveState:
        if self.path.exists():
            try:
                state = joblib.load(self.path)
                if (
                    isinstance(state, PriceAdaptiveState)
                    and state.schema_version == PRICE_ADAPTIVE_SCHEMA_VERSION
                ):
                    return state
            except Exception:
                pass
        return self._new_state(0)

    def _new_state(self, rebase_count: int) -> PriceAdaptiveState:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        seed = int(self.settings.section("model").get("random_state", 42))
        return PriceAdaptiveState(
            schema_version=PRICE_ADAPTIVE_SCHEMA_VERSION,
            champion_model_id=self.bundle.model_id,
            created_at=now,
            updated_at=now,
            direction_estimator=_direction_estimator(seed + 7001),
            return_estimator=_return_estimator(seed + 7002),
            rebase_count=rebase_count,
        )

    def synchronize(self, labeled_frame: pd.DataFrame) -> dict[str, Any]:
        frame = labeled_frame.copy().sort_values("open_time").reset_index(drop=True)
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        eligible = frame.dropna(
            subset=["target_up_h1", "future_return_h1"]
        ).copy()
        if self.state.last_trained_open_time:
            eligible = eligible[
                eligible["open_time"]
                > pd.Timestamp(self.state.last_trained_open_time)
            ]
        else:
            bootstrap_rows = max(
                200,
                int(self.config.get("online_bootstrap_rows", 2160)),
            )
            eligible = eligible.tail(bootstrap_rows)
        if eligible.empty:
            return self.summary()

        base = self.bundle.models[1].predict(
            eligible.reindex(columns=self.bundle.feature_columns)
        )
        observation_limit = max(
            100,
            int(self.config.get("online_observation_limit", 1500)),
        )
        for position, (_, row) in enumerate(eligible.iterrows()):
            base_probability = float(base["p_up"][position])
            base_return = float(base["general_return"][position])
            vector = price_vector(row, base_probability, base_return)
            target_up = int(row["target_up_h1"])
            target_return = float(row["future_return_h1"])

            # The observation is recorded before fitting this row. This keeps
            # the online comparison prequential rather than in-sample.
            if self.state.initialized:
                self.state.observations.append(
                    {
                        "open_time": pd.Timestamp(row["open_time"]).isoformat(),
                        "target_up": target_up,
                        "actual_return": target_return,
                        "base_probability_up": base_probability,
                        "online_probability_up": self._predict_probability(
                            vector, base_probability
                        ),
                        "base_return": base_return,
                        "online_return": self._predict_return(vector, base_return),
                    }
                )

            matrix = vector.reshape(1, -1)
            kwargs: dict[str, Any] = {}
            if not self.state.initialized:
                kwargs["classes"] = np.asarray([0, 1], dtype=int)
            self.state.direction_estimator.partial_fit(
                matrix,
                np.asarray([target_up], dtype=int),
                **kwargs,
            )
            self.state.return_estimator.partial_fit(
                matrix,
                np.asarray([target_return], dtype=float),
            )
            self.state.initialized = True
            self.state.samples_seen += 1
            self.state.last_trained_open_time = pd.Timestamp(
                row["open_time"]
            ).isoformat()

        self.state.observations = self.state.observations[-observation_limit:]
        self.state.updated_at = pd.Timestamp.now(tz="UTC").isoformat()
        self.save()
        return self.summary()

    def predict(
        self,
        row: pd.Series,
        base_probability_up: float,
        base_return: float,
    ) -> dict[str, Any]:
        base_probability_up = float(np.clip(base_probability_up, 0.05, 0.95))
        base_return = float(np.clip(base_return, -0.05, 0.05))
        vector = price_vector(row, base_probability_up, base_return)
        online_probability_up = self._predict_probability(
            vector, base_probability_up
        )
        online_return = self._predict_return(vector, base_return)
        metrics = self.metrics()
        direction_weight, return_weight = self._blend_weights(metrics)
        fused_probability_up = float(
            np.clip(
                (1.0 - direction_weight) * base_probability_up
                + direction_weight * online_probability_up,
                0.05,
                0.95,
            )
        )
        fused_return = float(
            np.clip(
                (1.0 - return_weight) * base_return
                + return_weight * online_return,
                -0.05,
                0.05,
            )
        )
        return {
            "source": (
                "BATCH_AND_ONLINE"
                if direction_weight > 0 or return_weight > 0
                else "BATCH_CHAMPION"
            ),
            "batch_probability_up": base_probability_up,
            "online_probability_up": online_probability_up,
            "fused_probability_up": fused_probability_up,
            "direction_blend_weight": direction_weight,
            "batch_return": base_return,
            "online_return": online_return,
            "fused_return": fused_return,
            "return_blend_weight": return_weight,
            "metrics": metrics,
            "samples_seen": self.state.samples_seen,
            "last_trained_open_time": self.state.last_trained_open_time,
            "champion_model_id": self.state.champion_model_id,
        }

    def _predict_probability(self, vector: np.ndarray, fallback: float) -> float:
        if not self.state.initialized:
            return float(np.clip(fallback, 0.05, 0.95))
        value = self.state.direction_estimator.predict_proba(
            vector.reshape(1, -1)
        )[0, 1]
        return float(np.clip(value, 0.05, 0.95))

    def _predict_return(self, vector: np.ndarray, fallback: float) -> float:
        if not self.state.initialized:
            return float(np.clip(fallback, -0.05, 0.05))
        value = self.state.return_estimator.predict(vector.reshape(1, -1))[0]
        return float(np.clip(value, -0.05, 0.05))

    def _blend_weights(self, metrics: dict[str, Any]) -> tuple[float, float]:
        samples = int(metrics.get("samples", 0))
        minimum_samples = max(
            1, int(self.config.get("online_minimum_samples", 168))
        )
        if samples < minimum_samples:
            return 0.0, 0.0
        maturity = min(1.0, samples / max(minimum_samples * 2, 1))

        max_direction = float(
            self.config.get("online_maximum_direction_weight", 0.35)
        )
        max_return = float(
            self.config.get("online_maximum_return_weight", 0.35)
        )
        minimum_brier_gain = float(
            self.config.get("online_minimum_brier_improvement", 0.0015)
        )
        minimum_accuracy_gain = float(
            self.config.get("online_minimum_accuracy_improvement", 0.002)
        )
        minimum_mae_gain = float(
            self.config.get("online_minimum_return_mae_improvement", 0.00002)
        )

        base_brier = _number(metrics.get("base_direction_brier"), 1.0)
        online_brier = _number(metrics.get("online_direction_brier"), 1.0)
        base_accuracy = _number(metrics.get("base_direction_accuracy"), 0.0)
        online_accuracy = _number(metrics.get("online_direction_accuracy"), 0.0)
        brier_gain = base_brier - online_brier
        accuracy_gain = online_accuracy - base_accuracy
        direction_safe = (
            brier_gain >= minimum_brier_gain
            and accuracy_gain >= minimum_accuracy_gain
        )
        direction_quality = min(
            1.0,
            brier_gain / max(minimum_brier_gain * 4.0, 1e-9),
        )
        direction_weight = (
            max_direction * maturity * direction_quality
            if direction_safe
            else 0.0
        )

        base_mae = _number(metrics.get("base_return_mae"), 1.0)
        online_mae = _number(metrics.get("online_return_mae"), 1.0)
        mae_gain = base_mae - online_mae
        return_safe = mae_gain >= minimum_mae_gain
        return_quality = min(
            1.0,
            mae_gain / max(minimum_mae_gain * 5.0, 1e-9),
        )
        return_weight = (
            max_return * maturity * return_quality
            if return_safe
            else 0.0
        )
        return float(direction_weight), float(return_weight)

    def metrics(self) -> dict[str, Any]:
        window = max(
            50, int(self.config.get("online_evaluation_window", 504))
        )
        observations = self.state.observations[-window:]
        if not observations:
            return {
                "samples": 0,
                "base_direction_brier": None,
                "online_direction_brier": None,
                "base_direction_accuracy": None,
                "online_direction_accuracy": None,
                "base_return_mae": None,
                "online_return_mae": None,
            }
        target = np.asarray([item["target_up"] for item in observations])
        base_p = np.asarray(
            [item["base_probability_up"] for item in observations], dtype=float
        )
        online_p = np.asarray(
            [item["online_probability_up"] for item in observations], dtype=float
        )
        actual_return = np.asarray(
            [item["actual_return"] for item in observations], dtype=float
        )
        base_return = np.asarray(
            [item["base_return"] for item in observations], dtype=float
        )
        online_return = np.asarray(
            [item["online_return"] for item in observations], dtype=float
        )
        return {
            "samples": int(len(observations)),
            "base_direction_brier": float(np.mean((base_p - target) ** 2)),
            "online_direction_brier": float(np.mean((online_p - target) ** 2)),
            "base_direction_accuracy": float(
                np.mean((base_p >= 0.5) == target)
            ),
            "online_direction_accuracy": float(
                np.mean((online_p >= 0.5) == target)
            ),
            "base_return_mae": float(
                np.mean(np.abs(base_return - actual_return))
            ),
            "online_return_mae": float(
                np.mean(np.abs(online_return - actual_return))
            ),
        }

    def summary(self) -> dict[str, Any]:
        metrics = self.metrics()
        direction_weight, return_weight = self._blend_weights(metrics)
        return {
            "schema_version": self.state.schema_version,
            "status": (
                "ACTIVE"
                if direction_weight > 0 or return_weight > 0
                else "SHADOW_LEARNING"
            ),
            "champion_model_id": self.state.champion_model_id,
            "samples_seen": self.state.samples_seen,
            "last_trained_open_time": self.state.last_trained_open_time,
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at,
            "rebase_count": self.state.rebase_count,
            "direction_blend_weight": direction_weight,
            "return_blend_weight": return_weight,
            "metrics": metrics,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        joblib.dump(self.state, temporary)
        temporary.replace(self.path)


def price_vector(
    row: pd.Series,
    base_probability_up: float,
    base_return: float,
) -> np.ndarray:
    probability = float(np.clip(base_probability_up, 1e-4, 1 - 1e-4))
    logit = np.log(probability / (1.0 - probability))
    values = np.asarray(
        [
            logit,
            base_return * 100.0,
            _series_number(row, "atr_pct") * 100.0,
            _series_number(row, "adx") / 50.0,
            _series_number(row, "rsi_centered"),
            _series_number(row, "price_vs_kama") * 100.0,
            _series_number(row, "return_1") * 100.0,
            _series_number(row, "return_3") * 100.0,
            _series_number(row, "realized_vol_24") * 100.0,
            _series_number(row, "volume_z_24"),
            _series_number(row, "event_score"),
            _series_number(row, "event_direction"),
            _series_number(row, "regime_code"),
            _series_number(row, "news_weighted_sent_6h"),
            _series_number(row, "news_relevance_6h"),
            min(_series_number(row, "news_age_hours"), 48.0) / 48.0,
            _series_number(row, "hour_sin"),
            _series_number(row, "hour_cos"),
            _series_number(row, "weekday_sin"),
            _series_number(row, "weekday_cos"),
        ],
        dtype=float,
    )
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=PRICE_VECTOR_LIMIT,
        neginf=-PRICE_VECTOR_LIMIT,
    )
    return np.clip(values, -PRICE_VECTOR_LIMIT, PRICE_VECTOR_LIMIT)


def _series_number(row: pd.Series, key: str) -> float:
    try:
        value = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)
