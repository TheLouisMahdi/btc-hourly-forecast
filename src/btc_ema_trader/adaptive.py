from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .costs import execution_cost_breakdown
from .model import HourlyModelBundle

ADAPTIVE_SCHEMA_VERSION = 1
ADAPTIVE_FEATURES = (
    "base_p_up",
    "base_general_return",
    "base_p_continuation",
    "base_p_tradeable",
    "base_event_return",
    "atr_pct",
    "adx",
    "rsi_centered",
    "price_vs_kama",
    "realized_vol_24",
    "volume_z_24",
    "event_score",
    "event_direction",
    "regime_code",
    "bars_since_event",
    "news_weighted_sent_6h",
    "news_relevance_6h",
    "news_age_hours",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)


def _binary_estimator(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        learning_rate="optimal",
        average=True,
        class_weight=None,
        random_state=seed,
    )


def _regression_estimator(seed: int) -> SGDRegressor:
    return SGDRegressor(
        loss="huber",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        average=True,
        random_state=seed,
    )


@dataclass
class OnlineBinaryLearner:
    scaler: StandardScaler = field(default_factory=StandardScaler)
    estimator: SGDClassifier = field(default_factory=lambda: _binary_estimator(1))
    initialized: bool = False
    samples_seen: int = 0

    def predict_probability(self, vector: np.ndarray, fallback: float) -> float:
        if not self.initialized:
            return float(np.clip(fallback, 1e-4, 1 - 1e-4))
        transformed = self.scaler.transform(vector.reshape(1, -1))
        return float(
            np.clip(
                self.estimator.predict_proba(transformed)[0, 1],
                1e-4,
                1 - 1e-4,
            )
        )

    def update(
        self,
        vector: np.ndarray,
        target: int,
        sample_weight: float = 1.0,
    ) -> None:
        matrix = vector.reshape(1, -1)
        self.scaler.partial_fit(matrix)
        transformed = self.scaler.transform(matrix)
        kwargs: dict[str, Any] = {
            "sample_weight": np.asarray([sample_weight], dtype=float)
        }
        if not self.initialized:
            kwargs["classes"] = np.asarray([0, 1], dtype=int)
        self.estimator.partial_fit(
            transformed,
            np.asarray([int(target)], dtype=int),
            **kwargs,
        )
        self.initialized = True
        self.samples_seen += 1


@dataclass
class OnlineRegressionLearner:
    scaler: StandardScaler = field(default_factory=StandardScaler)
    estimator: SGDRegressor = field(
        default_factory=lambda: _regression_estimator(1)
    )
    initialized: bool = False
    samples_seen: int = 0

    def predict(self, vector: np.ndarray, fallback: float) -> float:
        if not self.initialized:
            return float(fallback)
        transformed = self.scaler.transform(vector.reshape(1, -1))
        return float(self.estimator.predict(transformed)[0])

    def update(
        self,
        vector: np.ndarray,
        target: float,
        sample_weight: float = 1.0,
    ) -> None:
        matrix = vector.reshape(1, -1)
        self.scaler.partial_fit(matrix)
        transformed = self.scaler.transform(matrix)
        self.estimator.partial_fit(
            transformed,
            np.asarray([float(target)], dtype=float),
            sample_weight=np.asarray([sample_weight], dtype=float),
        )
        self.initialized = True
        self.samples_seen += 1


@dataclass
class AdaptiveHorizonModel:
    direction: OnlineBinaryLearner
    general_return: OnlineRegressionLearner
    continuation: OnlineBinaryLearner
    tradeability: OnlineBinaryLearner
    event_return: OnlineRegressionLearner

    @classmethod
    def create(cls, seed: int) -> "AdaptiveHorizonModel":
        return cls(
            direction=OnlineBinaryLearner(
                estimator=_binary_estimator(seed + 1)
            ),
            general_return=OnlineRegressionLearner(
                estimator=_regression_estimator(seed + 2)
            ),
            continuation=OnlineBinaryLearner(
                estimator=_binary_estimator(seed + 3)
            ),
            tradeability=OnlineBinaryLearner(
                estimator=_binary_estimator(seed + 4)
            ),
            event_return=OnlineRegressionLearner(
                estimator=_regression_estimator(seed + 5)
            ),
        )


@dataclass
class AdaptiveState:
    schema_version: int
    champion_model_id: str
    created_at: str
    updated_at: str
    horizons: dict[int, AdaptiveHorizonModel]
    last_trained_open_time: dict[int, str | None]
    observations: list[dict[str, Any]] = field(default_factory=list)
    active_horizons: list[int] = field(default_factory=list)
    suspended_horizons: list[int] = field(default_factory=list)
    rebase_count: int = 0


class AdaptiveEngine:
    def __init__(
        self,
        settings: Settings,
        bundle: HourlyModelBundle,
    ):
        self.settings = settings
        self.bundle = bundle
        self.config = settings.section("adaptive")
        self.enabled = bool(self.config.get("enabled", True))
        self.path = settings.path("adaptive_state")
        self.state = self._load_or_create()

    def _load_or_create(self) -> AdaptiveState:
        if self.enabled and self.path.exists():
            try:
                state = joblib.load(self.path)
                if (
                    isinstance(state, AdaptiveState)
                    and state.schema_version == ADAPTIVE_SCHEMA_VERSION
                ):
                    return state
            except Exception:
                pass
        return self._new_state(rebase_count=0)

    def _new_state(self, rebase_count: int) -> AdaptiveState:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        seed = int(self.settings.section("model").get("random_state", 42))
        horizons = {
            int(horizon): AdaptiveHorizonModel.create(
                seed + int(horizon) * 100
            )
            for horizon in self.bundle.horizons
        }
        return AdaptiveState(
            schema_version=ADAPTIVE_SCHEMA_VERSION,
            champion_model_id=self.bundle.model_id,
            created_at=now,
            updated_at=now,
            horizons=horizons,
            last_trained_open_time={
                int(horizon): None for horizon in self.bundle.horizons
            },
            rebase_count=rebase_count,
        )

    def synchronize(self, labeled_frame: pd.DataFrame) -> dict[str, Any]:
        if not self.enabled:
            return self.summary()
        if self.state.champion_model_id != self.bundle.model_id:
            self.state.champion_model_id = self.bundle.model_id
            self.state.observations = []
            self.state.active_horizons = []
            self.state.suspended_horizons = []
            self.state.rebase_count += 1

        frame = (
            labeled_frame.copy()
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        bootstrap_rows = max(
            100,
            int(self.config.get("bootstrap_rows", 1440)),
        )
        observation_limit = max(
            100,
            int(self.config.get("observation_limit", 1000)),
        )

        for horizon in self.bundle.horizons:
            horizon = int(horizon)
            target_column = f"target_up_h{horizon}"
            return_column = f"future_return_h{horizon}"
            eligible = frame.dropna(
                subset=[target_column, return_column]
            ).copy()
            last_time = self.state.last_trained_open_time.get(horizon)
            if last_time:
                eligible = eligible[
                    eligible["open_time"] > pd.Timestamp(last_time)
                ]
            else:
                eligible = eligible.tail(bootstrap_rows)
            if eligible.empty:
                continue

            base_outputs = self.bundle.models[horizon].predict(
                eligible.reindex(columns=self.bundle.feature_columns)
            )
            model = self.state.horizons[horizon]
            for position, (_, row) in enumerate(eligible.iterrows()):
                base = {
                    "p_up": float(base_outputs["p_up"][position]),
                    "general_return": float(
                        base_outputs["general_return"][position]
                    ),
                    "p_continuation": float(
                        base_outputs["p_continuation"][position]
                    ),
                    "p_tradeable": float(
                        base_outputs["p_tradeable"][position]
                    ),
                    "event_return": float(
                        base_outputs["event_return"][position]
                    ),
                }
                vector = adaptive_vector(row, base)
                target_up = int(row[target_column])
                target_return = float(row[return_column])

                if model.direction.initialized:
                    online_p_up = model.direction.predict_probability(
                        vector,
                        base["p_up"],
                    )
                    online_general_return = model.general_return.predict(
                        vector,
                        base["general_return"],
                    )
                    observation: dict[str, Any] = {
                        "open_time": pd.Timestamp(
                            row["open_time"]
                        ).isoformat(),
                        "horizon": horizon,
                        "target_up": target_up,
                        "base_p_up": base["p_up"],
                        "online_p_up": online_p_up,
                        "base_general_return": base["general_return"],
                        "online_general_return": online_general_return,
                        "actual_return": target_return,
                        "is_event": False,
                    }
                    continuation_column = (
                        f"event_continuation_h{horizon}"
                    )
                    tradeable_column = f"tradeable_h{horizon}"
                    event_return_column = (
                        f"event_gross_return_h{horizon}"
                    )
                    if (
                        continuation_column in row
                        and tradeable_column in row
                        and event_return_column in row
                        and pd.notna(row[continuation_column])
                        and pd.notna(row[tradeable_column])
                        and pd.notna(row[event_return_column])
                    ):
                        observation.update(
                            {
                                "is_event": True,
                                "target_continuation": int(
                                    row[continuation_column]
                                ),
                                "target_tradeable": int(
                                    row[tradeable_column]
                                ),
                                "actual_event_return": float(
                                    row[event_return_column]
                                ),
                                "base_p_continuation": base[
                                    "p_continuation"
                                ],
                                "online_p_continuation": model.continuation.predict_probability(
                                    vector,
                                    base["p_continuation"],
                                ),
                                "base_p_tradeable": base[
                                    "p_tradeable"
                                ],
                                "online_p_tradeable": model.tradeability.predict_probability(
                                    vector,
                                    base["p_tradeable"],
                                ),
                                "base_event_return": base[
                                    "event_return"
                                ],
                                "online_event_return": model.event_return.predict(
                                    vector,
                                    base["event_return"],
                                ),
                            }
                        )
                    self.state.observations.append(observation)

                sample_weight = self._sample_weight(row)
                model.direction.update(
                    vector,
                    target_up,
                    sample_weight,
                )
                model.general_return.update(
                    vector,
                    target_return,
                    sample_weight,
                )
                continuation_column = f"event_continuation_h{horizon}"
                tradeable_column = f"tradeable_h{horizon}"
                event_return_column = f"event_gross_return_h{horizon}"
                if (
                    continuation_column in row
                    and tradeable_column in row
                    and event_return_column in row
                    and pd.notna(row[continuation_column])
                    and pd.notna(row[tradeable_column])
                    and pd.notna(row[event_return_column])
                ):
                    event_weight = sample_weight * float(
                        self.config.get("event_weight", 2.0)
                    )
                    model.continuation.update(
                        vector,
                        int(row[continuation_column]),
                        event_weight,
                    )
                    model.tradeability.update(
                        vector,
                        int(row[tradeable_column]),
                        event_weight,
                    )
                    model.event_return.update(
                        vector,
                        float(row[event_return_column]),
                        event_weight,
                    )
                self.state.last_trained_open_time[horizon] = pd.Timestamp(
                    row["open_time"]
                ).isoformat()

        retained: list[dict[str, Any]] = []
        for horizon in self.bundle.horizons:
            horizon_observations = [
                item
                for item in self.state.observations
                if int(item.get("horizon", -1)) == int(horizon)
            ][-observation_limit:]
            retained.extend(horizon_observations)
        self.state.observations = sorted(
            retained,
            key=lambda item: (
                str(item.get("open_time", "")),
                int(item.get("horizon", 0)),
            ),
        )
        self._evaluate_horizons()
        self.state.updated_at = pd.Timestamp.now(tz="UTC").isoformat()
        self.save()
        return self.summary()

    def apply(
        self,
        row: pd.Series,
        base_prediction: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.enabled:
            return base_prediction, self.summary()
        corrected = deepcopy(base_prediction)
        shadow: dict[int, dict[str, float]] = {}
        active = {int(value) for value in self.state.active_horizons}
        blend_weight = float(self.config.get("blend_weight", 0.35))
        max_blend_weight = float(
            self.config.get("maximum_blend_weight", 0.55)
        )

        probabilities = {
            int(key): float(value)
            for key, value in corrected["probabilities"].items()
        }
        general_returns = {
            int(key): float(value)
            for key, value in corrected["returns"].items()
        }
        continuation = {
            int(key): float(value)
            for key, value in corrected["continuation"].items()
        }
        tradeability = {
            int(key): float(value)
            for key, value in corrected["tradeability"].items()
        }
        event_returns = {
            int(key): float(value)
            for key, value in corrected["event_returns"].items()
        }

        for horizon in self.bundle.horizons:
            horizon = int(horizon)
            model = self.state.horizons[horizon]
            base = {
                "p_up": probabilities[horizon],
                "general_return": general_returns[horizon],
                "p_continuation": continuation[horizon],
                "p_tradeable": tradeability[horizon],
                "event_return": event_returns[horizon],
            }
            vector = adaptive_vector(row, base)
            online = {
                "p_up": model.direction.predict_probability(
                    vector,
                    base["p_up"],
                ),
                "general_return": model.general_return.predict(
                    vector,
                    base["general_return"],
                ),
                "p_continuation": model.continuation.predict_probability(
                    vector,
                    base["p_continuation"],
                ),
                "p_tradeable": model.tradeability.predict_probability(
                    vector,
                    base["p_tradeable"],
                ),
                "event_return": model.event_return.predict(
                    vector,
                    base["event_return"],
                ),
            }
            horizon_metrics = self._metrics_for_horizon(horizon)
            direction_samples = int(
                horizon_metrics.get("direction_samples", 0)
            )
            minimum_samples = max(
                1,
                int(
                    self.config.get("minimum_direction_samples", 300)
                ),
            )
            maturity = min(1.0, direction_samples / minimum_samples)
            weight = (
                min(max_blend_weight, blend_weight * maturity)
                if horizon in active
                else 0.0
            )
            shadow[horizon] = {
                **online,
                "blend_weight": float(weight),
            }
            if weight > 0:
                probabilities[horizon] = _blend_probability(
                    base["p_up"],
                    online["p_up"],
                    weight,
                )
                general_returns[horizon] = _blend_value(
                    base["general_return"],
                    online["general_return"],
                    weight,
                )
                continuation[horizon] = _blend_probability(
                    base["p_continuation"],
                    online["p_continuation"],
                    weight,
                )
                tradeability[horizon] = _blend_probability(
                    base["p_tradeable"],
                    online["p_tradeable"],
                    weight,
                )
                event_returns[horizon] = _blend_value(
                    base["event_return"],
                    online["event_return"],
                    weight,
                )

        corrected["probabilities"] = probabilities
        corrected["returns"] = general_returns
        corrected["continuation"] = continuation
        corrected["tradeability"] = tradeability
        corrected["event_returns"] = event_returns
        corrected = _recalculate_prediction(
            corrected,
            self.bundle,
            self.settings,
        )
        summary = self.summary()
        summary["shadow_predictions"] = shadow
        summary["decision_source"] = (
            "ADAPTIVE_BLEND" if active else "BATCH_CHAMPION"
        )
        return corrected, summary

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        joblib.dump(self.state, temporary)
        temporary.replace(self.path)

    def summary(self) -> dict[str, Any]:
        metrics = {
            str(horizon): self._metrics_for_horizon(int(horizon))
            for horizon in self.bundle.horizons
        }
        active = sorted(
            int(value) for value in self.state.active_horizons
        )
        suspended = sorted(
            int(value) for value in self.state.suspended_horizons
        )
        if not self.enabled:
            status = "DISABLED"
        elif active and len(active) == len(self.bundle.horizons):
            status = "ACTIVE"
        elif active:
            status = "PARTIALLY_ACTIVE"
        elif suspended:
            status = "SUSPENDED"
        else:
            status = "SHADOW"
        return {
            "schema_version": self.state.schema_version,
            "status": status,
            "champion_model_id": self.state.champion_model_id,
            "active_horizons": active,
            "suspended_horizons": suspended,
            "last_trained_open_time": self.state.last_trained_open_time,
            "rebase_count": self.state.rebase_count,
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at,
            "feature_count": len(ADAPTIVE_FEATURES),
            "metrics": metrics,
        }

    def _sample_weight(self, row: pd.Series) -> float:
        recency_multiplier = float(
            self.config.get("new_sample_weight", 1.0)
        )
        return max(0.01, recency_multiplier)

    def _evaluate_horizons(self) -> None:
        active = {int(value) for value in self.state.active_horizons}
        suspended = {
            int(value) for value in self.state.suspended_horizons
        }
        minimum_direction = int(
            self.config.get("minimum_direction_samples", 300)
        )
        minimum_events = int(
            self.config.get("minimum_event_samples", 60)
        )
        minimum_improvement = float(
            self.config.get("minimum_brier_improvement", 0.003)
        )
        maximum_accuracy_regression = float(
            self.config.get("maximum_accuracy_regression", 0.015)
        )
        suspension_margin = float(
            self.config.get("suspension_brier_margin", 0.02)
        )

        for horizon in self.bundle.horizons:
            horizon = int(horizon)
            metrics = self._metrics_for_horizon(horizon)
            ready = (
                metrics.get("direction_samples", 0) >= minimum_direction
                and metrics.get("event_samples", 0) >= minimum_events
            )
            base_direction_brier = _metric(
                metrics,
                "base_direction_brier",
                1.0,
            )
            online_direction_brier = _metric(
                metrics,
                "online_direction_brier",
                1.0,
            )
            base_event_brier = _metric(
                metrics,
                "base_event_brier",
                1.0,
            )
            online_event_brier = _metric(
                metrics,
                "online_event_brier",
                1.0,
            )
            base_direction_accuracy = _metric(
                metrics,
                "base_direction_accuracy",
                0.0,
            )
            online_direction_accuracy = _metric(
                metrics,
                "online_direction_accuracy",
                0.0,
            )
            direction_improved = (
                online_direction_brier
                <= base_direction_brier - minimum_improvement
            )
            event_improved = (
                online_event_brier
                <= base_event_brier - minimum_improvement
            )
            accuracy_safe = (
                online_direction_accuracy
                >= base_direction_accuracy - maximum_accuracy_regression
            )
            degraded = (
                online_direction_brier
                > base_direction_brier + suspension_margin
                or online_event_brier
                > base_event_brier + suspension_margin
            )
            if horizon in active and degraded:
                active.discard(horizon)
                suspended.add(horizon)
            elif ready and direction_improved and event_improved and accuracy_safe:
                active.add(horizon)
                suspended.discard(horizon)
            elif (
                horizon in suspended
                and ready
                and direction_improved
                and event_improved
                and accuracy_safe
            ):
                active.add(horizon)
                suspended.discard(horizon)

        self.state.active_horizons = sorted(active)
        self.state.suspended_horizons = sorted(suspended)

    def _metrics_for_horizon(self, horizon: int) -> dict[str, Any]:
        window = max(
            50,
            int(self.config.get("evaluation_window", 500)),
        )
        observations = [
            item
            for item in self.state.observations
            if int(item.get("horizon", -1)) == horizon
        ][-window:]
        if not observations:
            return {
                "direction_samples": 0,
                "event_samples": 0,
                "base_direction_brier": None,
                "online_direction_brier": None,
                "base_direction_accuracy": None,
                "online_direction_accuracy": None,
                "base_event_brier": None,
                "online_event_brier": None,
                "base_return_mae": None,
                "online_return_mae": None,
            }

        target = np.asarray(
            [int(item["target_up"]) for item in observations],
            dtype=float,
        )
        base_p = np.asarray(
            [float(item["base_p_up"]) for item in observations],
            dtype=float,
        )
        online_p = np.asarray(
            [float(item["online_p_up"]) for item in observations],
            dtype=float,
        )
        actual_return = np.asarray(
            [float(item["actual_return"]) for item in observations],
            dtype=float,
        )
        base_return = np.asarray(
            [
                float(item["base_general_return"])
                for item in observations
            ],
            dtype=float,
        )
        online_return = np.asarray(
            [
                float(item["online_general_return"])
                for item in observations
            ],
            dtype=float,
        )
        event_observations = [
            item for item in observations if bool(item.get("is_event"))
        ]
        base_event_brier: float | None = None
        online_event_brier: float | None = None
        if event_observations:
            continuation_target = np.asarray(
                [
                    int(item["target_continuation"])
                    for item in event_observations
                ],
                dtype=float,
            )
            tradeability_target = np.asarray(
                [
                    int(item["target_tradeable"])
                    for item in event_observations
                ],
                dtype=float,
            )
            base_event_brier = float(
                0.5
                * (
                    np.mean(
                        (
                            np.asarray(
                                [
                                    item["base_p_continuation"]
                                    for item in event_observations
                                ]
                            )
                            - continuation_target
                        )
                        ** 2
                    )
                    + np.mean(
                        (
                            np.asarray(
                                [
                                    item["base_p_tradeable"]
                                    for item in event_observations
                                ]
                            )
                            - tradeability_target
                        )
                        ** 2
                    )
                )
            )
            online_event_brier = float(
                0.5
                * (
                    np.mean(
                        (
                            np.asarray(
                                [
                                    item["online_p_continuation"]
                                    for item in event_observations
                                ]
                            )
                            - continuation_target
                        )
                        ** 2
                    )
                    + np.mean(
                        (
                            np.asarray(
                                [
                                    item["online_p_tradeable"]
                                    for item in event_observations
                                ]
                            )
                            - tradeability_target
                        )
                        ** 2
                    )
                )
            )
        return {
            "direction_samples": len(observations),
            "event_samples": len(event_observations),
            "base_direction_brier": float(
                np.mean((base_p - target) ** 2)
            ),
            "online_direction_brier": float(
                np.mean((online_p - target) ** 2)
            ),
            "base_direction_accuracy": float(
                np.mean((base_p >= 0.5) == target)
            ),
            "online_direction_accuracy": float(
                np.mean((online_p >= 0.5) == target)
            ),
            "base_event_brier": base_event_brier,
            "online_event_brier": online_event_brier,
            "base_return_mae": float(
                np.mean(np.abs(base_return - actual_return))
            ),
            "online_return_mae": float(
                np.mean(np.abs(online_return - actual_return))
            ),
        }


def adaptive_vector(
    row: pd.Series,
    base: dict[str, float],
) -> np.ndarray:
    raw = {
        "base_p_up": base.get("p_up", 0.5),
        "base_general_return": base.get("general_return", 0.0) * 100.0,
        "base_p_continuation": base.get("p_continuation", 0.5),
        "base_p_tradeable": base.get("p_tradeable", 0.5),
        "base_event_return": base.get("event_return", 0.0) * 100.0,
        "atr_pct": row.get("atr_pct", 0.0) * 100.0,
        "adx": row.get("adx", 0.0) / 100.0,
        "rsi_centered": row.get("rsi_centered", 0.0),
        "price_vs_kama": row.get("price_vs_kama", 0.0) * 100.0,
        "realized_vol_24": row.get("realized_vol_24", 0.0) * 100.0,
        "volume_z_24": row.get("volume_z_24", 0.0),
        "event_score": row.get("event_score", 0.0),
        "event_direction": row.get("event_direction", 0.0),
        "regime_code": row.get("regime_code", 0.0),
        "bars_since_event": min(
            float(row.get("bars_since_event", 24.0) or 24.0),
            72.0,
        )
        / 24.0,
        "news_weighted_sent_6h": row.get(
            "news_weighted_sent_6h",
            0.0,
        ),
        "news_relevance_6h": row.get("news_relevance_6h", 0.0),
        "news_age_hours": min(
            float(row.get("news_age_hours", 72.0) or 72.0),
            168.0,
        )
        / 24.0,
        "hour_sin": row.get("hour_sin", 0.0),
        "hour_cos": row.get("hour_cos", 0.0),
        "weekday_sin": row.get("weekday_sin", 0.0),
        "weekday_cos": row.get("weekday_cos", 0.0),
    }
    values = np.asarray(
        [_finite(raw[name]) for name in ADAPTIVE_FEATURES],
        dtype=float,
    )
    return np.clip(values, -8.0, 8.0)


def _recalculate_prediction(
    prediction: dict[str, Any],
    bundle: HourlyModelBundle,
    settings: Settings,
) -> dict[str, Any]:
    probabilities = {
        int(key): float(value)
        for key, value in prediction["probabilities"].items()
    }
    general_returns = {
        int(key): float(value)
        for key, value in prediction["returns"].items()
    }
    continuation = {
        int(key): float(value)
        for key, value in prediction["continuation"].items()
    }
    tradeability = {
        int(key): float(value)
        for key, value in prediction["tradeability"].items()
    }
    event_returns = {
        int(key): float(value)
        for key, value in prediction["event_returns"].items()
    }
    event_direction = int(prediction.get("event_direction", 0))
    qualified = [
        int(horizon)
        for horizon in bundle.qualification.get(
            "qualified_horizons",
            [],
        )
    ]
    candidates = qualified or [int(horizon) for horizon in bundle.horizons]
    stress_cost = (
        execution_cost_breakdown(settings.section("strategy"))[
            "stress_cost_bps"
        ]
        / 10_000
    )

    scores: dict[int, float] = {}
    for horizon in bundle.horizons:
        horizon = int(horizon)
        fold_consistency = float(
            bundle.metrics.get(str(horizon), {}).get(
                "positive_fold_fraction",
                0.0,
            )
        )
        quality = 0.25 + 0.75 * fold_consistency
        predicted_net = event_returns[horizon] - stress_cost
        scores[horizon] = (
            continuation[horizon]
            * tradeability[horizon]
            * max(predicted_net, 0.0)
            * quality
        )
    if any(scores[horizon] > 0 for horizon in candidates):
        selected_horizon = max(
            candidates,
            key=lambda horizon: scores[horizon],
        )
    else:
        selected_horizon = max(
            candidates,
            key=lambda horizon: (
                continuation[horizon] * tradeability[horizon]
            ),
        )

    score = sum(
        bundle.horizon_weights[int(horizon)]
        * (2 * probabilities[int(horizon)] - 1)
        for horizon in bundle.horizons
    )
    direction = "UP" if score >= 0 else "DOWN"
    confidence = float(0.5 + abs(score) / 2)
    votes = [
        "UP" if probabilities[int(horizon)] >= 0.5 else "DOWN"
        for horizon in bundle.horizons
    ]
    agreement = max(votes.count("UP"), votes.count("DOWN")) / len(
        votes
    )
    event_agreement = float(
        np.mean(
            [
                continuation[int(horizon)] >= 0.5
                for horizon in bundle.horizons
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
    absolute_event_returns = {
        int(horizon): (
            float(event_returns[int(horizon)] * event_direction)
            if event_direction
            else 0.0
        )
        for horizon in bundle.horizons
    }
    prediction.update(
        {
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
            "absolute_event_returns": absolute_event_returns,
            "horizon_scores": scores,
            "selected_horizon": int(selected_horizon),
            "expected_return": absolute_event_returns[selected_horizon],
            "expected_event_aligned_return": event_returns[
                selected_horizon
            ],
        }
    )
    return prediction


def _blend_probability(
    base: float,
    online: float,
    weight: float,
) -> float:
    return float(
        np.clip(
            (1.0 - weight) * base + weight * online,
            1e-4,
            1 - 1e-4,
        )
    )


def _blend_value(base: float, online: float, weight: float) -> float:
    return float((1.0 - weight) * base + weight * online)


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _metric(
    metrics: dict[str, Any],
    key: str,
    default: float,
) -> float:
    value = metrics.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default
