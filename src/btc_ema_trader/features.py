from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .costs import execution_cost_breakdown
from .market_structure import build_market_structure


@dataclass(frozen=True)
class FeatureSet:
    frame: pd.DataFrame
    feature_columns: list[str]
    horizons: list[int]


def build_feature_set(
    candles: pd.DataFrame,
    news: pd.DataFrame,
    settings: Settings,
    include_labels: bool = True,
) -> FeatureSet:
    if candles.empty:
        raise ValueError("No candles available")

    cfg = settings.section("features")
    model_cfg = settings.section("model")
    strategy_cfg = settings.section("strategy")
    structure_cfg = settings.section("structure")
    horizons = [
        int(horizon)
        for horizon in model_cfg.get(
            "horizons_hours",
            [1, 3, 6],
        )
    ]

    frame = (
        candles.copy()
        .sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )
    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        utc=True,
    )
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]

    kama_er = int(cfg.get("kama_er_period", 10))
    kama_fast = int(cfg.get("kama_fast_period", 2))
    kama_slow = int(cfg.get("kama_slow_period", 30))
    frame["kama"] = _kama(
        close,
        kama_er,
        kama_fast,
        kama_slow,
    )
    frame["kama_slope_1"] = frame["kama"].pct_change()
    frame["kama_slope_3"] = frame["kama"].pct_change(3) / 3
    frame["kama_slope_12"] = frame["kama"].pct_change(12) / 12
    frame["price_vs_kama"] = close / frame["kama"] - 1

    for span in (24, 72, 168, 336):
        ema = close.ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()
        frame[f"ema_{span}"] = ema
        frame[f"price_vs_ema_{span}"] = close / ema - 1
        frame[f"ema_{span}_slope_6"] = ema.pct_change(6) / 6

    atr_period = int(cfg.get("atr_period", 14))
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = true_range.ewm(
        alpha=1 / atr_period,
        adjust=False,
        min_periods=atr_period,
    ).mean()
    frame["atr_pct"] = frame["atr"] / close
    frame["atr_z_72"] = _rolling_zscore(
        frame["atr_pct"],
        72,
    )
    frame["atr_percentile_168"] = frame[
        "atr_pct"
    ].rolling(168).apply(
        _last_percentile_rank,
        raw=True,
    )

    adx_period = int(cfg.get("adx_period", 14))
    high_change = high.diff()
    low_change = -low.diff()
    plus_dm = high_change.where(
        (high_change > low_change) & (high_change > 0),
        0.0,
    )
    minus_dm = low_change.where(
        (low_change > high_change) & (low_change > 0),
        0.0,
    )
    atr_wilder = true_range.ewm(
        alpha=1 / adx_period,
        adjust=False,
        min_periods=adx_period,
    ).mean()
    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / adx_period,
            adjust=False,
            min_periods=adx_period,
        ).mean()
        / atr_wilder
    )
    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / adx_period,
            adjust=False,
            min_periods=adx_period,
        ).mean()
        / atr_wilder
    )
    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)
    frame["plus_di"] = plus_di
    frame["minus_di"] = minus_di
    frame["adx"] = dx.ewm(
        alpha=1 / adx_period,
        adjust=False,
        min_periods=adx_period,
    ).mean()
    frame["di_spread"] = (plus_di - minus_di) / 100

    donchian_period = int(cfg.get("donchian_period", 20))
    frame["donchian_high"] = high.shift(1).rolling(
        donchian_period
    ).max()
    frame["donchian_low"] = low.shift(1).rolling(
        donchian_period
    ).min()
    frame["donchian_mid"] = (
        frame["donchian_high"] + frame["donchian_low"]
    ) / 2
    frame["donchian_position"] = (
        close - frame["donchian_low"]
    ) / (
        frame["donchian_high"] - frame["donchian_low"]
    ).replace(0, np.nan)
    frame["donchian_width_pct"] = (
        frame["donchian_high"] - frame["donchian_low"]
    ) / close

    bb_period = int(cfg.get("bollinger_period", 20))
    bb_std = float(cfg.get("bollinger_std", 2.0))
    frame["bb_mid"] = close.rolling(bb_period).mean()
    bb_sigma = close.rolling(bb_period).std(ddof=0)
    frame["bb_upper"] = frame["bb_mid"] + bb_std * bb_sigma
    frame["bb_lower"] = frame["bb_mid"] - bb_std * bb_sigma
    frame["bb_width"] = (
        frame["bb_upper"] - frame["bb_lower"]
    ) / frame["bb_mid"]
    squeeze_lookback = int(
        cfg.get("squeeze_lookback", 120)
    )
    frame["bb_width_percentile"] = frame[
        "bb_width"
    ].rolling(squeeze_lookback).apply(
        _last_percentile_rank,
        raw=True,
    )

    rsi_period = int(cfg.get("rsi_period", 14))
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(
        alpha=1 / rsi_period,
        adjust=False,
        min_periods=rsi_period,
    ).mean()
    losses = (-delta.clip(upper=0)).ewm(
        alpha=1 / rsi_period,
        adjust=False,
        min_periods=rsi_period,
    ).mean()
    rs = gains / losses.replace(0, np.nan)
    frame["rsi14"] = 100 - 100 / (1 + rs)
    frame["rsi_centered"] = (frame["rsi14"] - 50) / 50

    for lookback in (
        1,
        2,
        3,
        6,
        12,
        24,
        48,
        72,
        120,
        168,
    ):
        frame[f"return_{lookback}"] = close.pct_change(
            lookback
        )
    log_return = np.log(close).diff()
    frame["log_return_1"] = log_return
    for lookback in (3, 6, 12, 24, 48, 72, 168):
        frame[f"realized_vol_{lookback}"] = (
            log_return.rolling(lookback).std()
            * np.sqrt(lookback)
        )

    frame["range_pct"] = (high - low) / close
    frame["body_pct"] = (close - open_) / open_
    frame["body_atr"] = (close - open_) / frame["atr"]
    frame["upper_wick_pct"] = (
        high - frame[["open", "close"]].max(axis=1)
    ) / close
    frame["lower_wick_pct"] = (
        frame[["open", "close"]].min(axis=1) - low
    ) / close
    frame["close_location"] = (
        close - low
    ) / (high - low).replace(0, np.nan)
    frame["range_expansion_atr"] = (
        high - low
    ) / frame["atr"]

    volume_log = np.log1p(frame["volume"])
    frame["volume_change_1"] = volume_log.diff()
    frame["volume_z_24"] = _rolling_zscore(
        volume_log,
        24,
    )
    frame["volume_z_72"] = _rolling_zscore(
        volume_log,
        72,
    )
    frame["volume_trend_24_72"] = (
        volume_log.rolling(24).mean()
        - volume_log.rolling(72).mean()
    )
    if "quote_volume" in frame:
        frame["quote_volume_z_24"] = _rolling_zscore(
            np.log1p(frame["quote_volume"]),
            24,
        )
    else:
        frame["quote_volume_z_24"] = 0.0
    if "trades" in frame:
        frame["trades_z_24"] = _rolling_zscore(
            np.log1p(frame["trades"].fillna(0)),
            24,
        )
    else:
        frame["trades_z_24"] = 0.0

    hours = frame["open_time"].dt.hour
    weekdays = frame["open_time"].dt.dayofweek
    frame["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    frame["weekday_sin"] = np.sin(
        2 * np.pi * weekdays / 7
    )
    frame["weekday_cos"] = np.cos(
        2 * np.pi * weekdays / 7
    )

    frame = build_market_structure(frame, settings)

    trend_adx = float(
        cfg.get("trend_adx_threshold", 18.0)
    )
    long_term_up = (
        (close > frame["ema_168"])
        & (frame["ema_168_slope_6"] > 0)
        & (frame["adx"] >= trend_adx)
    )
    long_term_down = (
        (close < frame["ema_168"])
        & (frame["ema_168_slope_6"] < 0)
        & (frame["adx"] >= trend_adx)
    )
    compression = (
        (frame["triangle_code"] != 0)
        | (
            frame["bb_width_percentile"]
            <= float(
                cfg.get("squeeze_percentile", 0.20)
            )
        )
    )
    frame["regime_code"] = np.select(
        [long_term_up, long_term_down],
        [1, -1],
        default=0,
    ).astype(int)
    frame["regime"] = np.select(
        [long_term_up, long_term_down, compression],
        [
            "STRUCTURE_UP",
            "STRUCTURE_DOWN",
            "COMPRESSION",
        ],
        default="RANGE",
    )
    frame["event_continuation_bias"] = (
        frame["event_direction"] * frame["regime_code"]
    )

    bars_since_event: list[float] = []
    active_event_direction: list[int] = []
    last_index: int | None = None
    last_direction = 0
    for index, direction in enumerate(
        frame["event_direction"].astype(int).tolist()
    ):
        if direction != 0:
            last_index = index
            last_direction = direction
        bars_since_event.append(
            np.nan
            if last_index is None
            else float(index - last_index)
        )
        active_event_direction.append(last_direction)
    frame["bars_since_event"] = bars_since_event
    frame["active_event_direction"] = active_event_direction

    news_features = aggregate_news_hourly(
        news,
        frame["open_time"],
        settings,
    )
    frame = frame.merge(
        news_features,
        on="open_time",
        how="left",
    )
    news_columns = [
        column
        for column in frame.columns
        if column.startswith("news_")
    ]
    frame[news_columns] = frame[news_columns].fillna(0.0)
    frame["news_shock"] = (
        (
            frame.get("news_count_1h", 0)
            >= float(cfg.get("news_shock_min_count", 3))
        )
        & (
            frame.get("news_relevance_3h", 0)
            >= float(
                cfg.get("news_shock_min_relevance", 0.55)
            )
        )
    ).astype(int)

    if include_labels:
        _attach_labels(
            frame,
            horizons,
            strategy_cfg,
            structure_cfg,
        )

    excluded = {
        "provider",
        "symbol",
        "open_time",
        "fetched_at",
        "closed",
        "open",
        "high",
        "low",
        "close",
        "event_id",
        "event_type",
        "regime",
        "triangle_type",
        "breakout_source",
        "kama",
        "donchian_high",
        "donchian_low",
        "donchian_mid",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "atr",
        "structure_resistance",
        "structure_support",
        "triangle_upper",
        "triangle_lower",
        "breakout_level",
        "breakout_invalidation_level",
    }
    for horizon in horizons:
        excluded.update(
            {
                f"entry_price_h{horizon}",
                f"future_return_h{horizon}",
                f"target_up_h{horizon}",
                f"mfe_h{horizon}",
                f"mae_h{horizon}",
                f"barrier_outcome_h{horizon}",
                f"breakout_hold_h{horizon}",
                f"breakout_success_h{horizon}",
                f"false_breakout_h{horizon}",
                f"neutral_breakout_h{horizon}",
                f"event_continuation_h{horizon}",
                f"tradeable_h{horizon}",
                f"event_gross_return_h{horizon}",
                f"event_net_return_h{horizon}",
            }
        )
    feature_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return FeatureSet(
        frame=frame,
        feature_columns=feature_columns,
        horizons=horizons,
    )


def _attach_labels(
    frame: pd.DataFrame,
    horizons: list[int],
    strategy_cfg: dict[str, Any],
    structure_cfg: dict[str, Any],
) -> None:
    costs = execution_cost_breakdown(strategy_cfg)
    base_cost = costs["base_cost_bps"] / 10_000
    profit_buffer = costs["profit_buffer_bps"] / 10_000
    target_atr = float(
        strategy_cfg.get(
            "label_target_atr_multiplier",
            1.35,
        )
    )
    stop_atr = float(
        strategy_cfg.get(
            "label_stop_atr_multiplier",
            1.0,
        )
    )
    hold_buffer_atr = float(
        structure_cfg.get("label_hold_buffer_atr", 0.05)
    )

    for horizon in horizons:
        entry = frame["open"].shift(-1)
        future_close = frame["close"].shift(-horizon)
        raw_return = future_close / entry - 1
        frame[f"entry_price_h{horizon}"] = entry
        frame[f"future_return_h{horizon}"] = raw_return
        frame[f"target_up_h{horizon}"] = (
            raw_return > 0
        ).astype(float)
        frame.loc[
            raw_return.isna(),
            f"target_up_h{horizon}",
        ] = np.nan

        mfe, mae, barrier = _path_labels(
            frame,
            horizon,
            target_atr,
            stop_atr,
        )
        frame[f"mfe_h{horizon}"] = mfe
        frame[f"mae_h{horizon}"] = mae
        frame[f"barrier_outcome_h{horizon}"] = barrier

        event_direction = frame[
            "event_direction"
        ].astype(float)
        event_gross = event_direction * raw_return
        atr_move = frame["atr"] / entry
        event_gross = event_gross.where(
            barrier != 1,
            target_atr * atr_move,
        )
        event_gross = event_gross.where(
            barrier != -1,
            -stop_atr * atr_move,
        )
        event_net = event_gross - base_cost

        level = frame["breakout_level"]
        hold_buffer = frame["atr"] * hold_buffer_atr
        hold = np.where(
            event_direction > 0,
            future_close > level + hold_buffer,
            np.where(
                event_direction < 0,
                future_close < level - hold_buffer,
                False,
            ),
        )
        hold = pd.Series(
            hold,
            index=frame.index,
            dtype=float,
        )
        success = (
            hold.astype(bool)
            & (barrier != -1)
            & (event_gross > 0)
        )
        reentered_structure = np.where(
            event_direction > 0,
            future_close <= level,
            np.where(
                event_direction < 0,
                future_close >= level,
                False,
            ),
        )
        reentered_structure = pd.Series(
            reentered_structure,
            index=frame.index,
            dtype=bool,
        )
        false_breakout = (
            (reentered_structure | (barrier == -1))
            & (event_direction != 0)
        )
        neutral_breakout = (
            (~success)
            & (~false_breakout)
            & (event_direction != 0)
        )
        tradeable = success & (event_net >= profit_buffer)

        missing = (
            (event_direction == 0)
            | raw_return.isna()
            | pd.isna(barrier)
            | level.isna()
        )
        continuation = success.astype(float)
        tradeable_float = tradeable.astype(float)
        false_breakout_float = false_breakout.astype(float)
        neutral_breakout_float = neutral_breakout.astype(float)
        hold[missing] = np.nan
        continuation[missing] = np.nan
        tradeable_float[missing] = np.nan
        false_breakout_float[missing] = np.nan
        neutral_breakout_float[missing] = np.nan
        event_gross[missing] = np.nan
        event_net[missing] = np.nan

        frame[f"breakout_hold_h{horizon}"] = hold
        frame[f"breakout_success_h{horizon}"] = continuation
        frame[f"false_breakout_h{horizon}"] = (
            false_breakout_float
        )
        frame[f"neutral_breakout_h{horizon}"] = (
            neutral_breakout_float
        )
        frame[f"event_continuation_h{horizon}"] = continuation
        frame[f"tradeable_h{horizon}"] = tradeable_float
        frame[f"event_gross_return_h{horizon}"] = event_gross
        frame[f"event_net_return_h{horizon}"] = event_net


def _path_labels(
    frame: pd.DataFrame,
    horizon: int,
    target_atr_multiplier: float,
    stop_atr_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(frame)
    mfe = np.full(count, np.nan)
    mae = np.full(count, np.nan)
    barrier = np.full(count, np.nan)
    for index in range(count - horizon):
        direction = int(
            frame.iloc[index]["event_direction"]
        )
        if direction == 0:
            continue
        entry = float(frame.iloc[index + 1]["open"])
        atr = float(frame.iloc[index]["atr"])
        invalidation = float(
            frame.iloc[index].get(
                "breakout_invalidation_level",
                np.nan,
            )
        )
        if (
            not np.isfinite(entry)
            or not np.isfinite(atr)
            or entry <= 0
            or atr <= 0
        ):
            continue
        highs = frame.iloc[
            index + 1 : index + horizon + 1
        ]["high"].to_numpy(dtype=float)
        lows = frame.iloc[
            index + 1 : index + horizon + 1
        ]["low"].to_numpy(dtype=float)
        if direction > 0:
            favorable = highs / entry - 1
            adverse = entry / lows - 1
            target_price = (
                entry + target_atr_multiplier * atr
            )
            stop_price = (
                invalidation
                if np.isfinite(invalidation)
                else entry - stop_atr_multiplier * atr
            )
            target_hits = highs >= target_price
            stop_hits = lows <= stop_price
        else:
            favorable = entry / lows - 1
            adverse = highs / entry - 1
            target_price = (
                entry - target_atr_multiplier * atr
            )
            stop_price = (
                invalidation
                if np.isfinite(invalidation)
                else entry + stop_atr_multiplier * atr
            )
            target_hits = lows <= target_price
            stop_hits = highs >= stop_price
        mfe[index] = float(np.nanmax(favorable))
        mae[index] = float(np.nanmax(adverse))
        outcome = 0.0
        for bar_index in range(len(highs)):
            if bool(stop_hits[bar_index]):
                outcome = -1.0
                break
            if bool(target_hits[bar_index]):
                outcome = 1.0
                break
        barrier[index] = outcome
    return mfe, mae, barrier


def aggregate_news_hourly(
    news: pd.DataFrame,
    candle_times: pd.Series,
    settings: Settings,
) -> pd.DataFrame:
    open_times = pd.to_datetime(
        candle_times,
        utc=True,
    )
    decision_times = pd.DatetimeIndex(
        open_times + pd.Timedelta(hours=1),
        name="decision_time",
    )
    base = pd.DataFrame(index=decision_times)
    windows = [
        int(value)
        for value in settings.section("features").get(
            "news_windows_hours",
            [1, 3, 6, 12, 24],
        )
    ]
    if news.empty:
        for window in windows:
            for suffix in (
                "count",
                "sent_mean",
                "sent_sum",
                "weighted_sent",
                "relevance",
                "negative_share",
            ):
                base[f"news_{suffix}_{window}h"] = 0.0
        base["news_age_hours"] = 999.0
        base["news_available"] = 0.0
        base["open_time"] = open_times.to_numpy()
        return base.reset_index(drop=True)

    articles = news.copy()
    articles["published_at"] = pd.to_datetime(
        articles["published_at"],
        utc=True,
    )
    if "first_seen_at" in articles:
        articles["first_seen_at"] = pd.to_datetime(
            articles["first_seen_at"],
            utc=True,
        )
        articles["available_at"] = articles[
            ["published_at", "first_seen_at"]
        ].max(axis=1)
    else:
        articles["available_at"] = articles["published_at"]
    articles["sentiment"] = pd.to_numeric(
        articles["sentiment"],
        errors="coerce",
    ).fillna(0.0)
    articles["relevance"] = pd.to_numeric(
        articles["relevance"],
        errors="coerce",
    ).fillna(0.0)
    articles = articles.sort_values("available_at")
    articles["decision_time"] = articles[
        "available_at"
    ].dt.ceil("h")
    articles["weighted_sent"] = articles[
        "sentiment"
    ] * (0.25 + articles["relevance"])
    articles["negative"] = (
        articles["sentiment"] < -0.25
    ).astype(float)
    hourly = articles.groupby("decision_time").agg(
        count=("sentiment", "size"),
        sent_sum=("sentiment", "sum"),
        weighted_sum=("weighted_sent", "sum"),
        relevance_sum=("relevance", "sum"),
        negative_count=("negative", "sum"),
    ).reindex(decision_times, fill_value=0.0)
    for window in windows:
        rolling = hourly.rolling(
            window=window,
            min_periods=1,
        ).sum()
        count = rolling["count"].replace(0, np.nan)
        base[f"news_count_{window}h"] = rolling[
            "count"
        ].astype(float)
        base[f"news_sent_sum_{window}h"] = rolling[
            "sent_sum"
        ].astype(float)
        base[f"news_sent_mean_{window}h"] = (
            rolling["sent_sum"] / count
        ).fillna(0.0)
        base[f"news_weighted_sent_{window}h"] = (
            rolling["weighted_sum"] / count
        ).fillna(0.0)
        base[f"news_relevance_{window}h"] = (
            rolling["relevance_sum"] / count
        ).fillna(0.0)
        base[f"news_negative_share_{window}h"] = (
            rolling["negative_count"] / count
        ).fillna(0.0)

    decision_frame = pd.DataFrame(
        {"decision_time": decision_times}
    ).sort_values("decision_time")
    latest = pd.merge_asof(
        decision_frame,
        articles[["available_at"]].sort_values(
            "available_at"
        ),
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
    )
    age = (
        latest["decision_time"] - latest["available_at"]
    ).dt.total_seconds().div(3600)
    base["news_age_hours"] = age.fillna(999.0).to_numpy()
    base["news_available"] = (
        latest["available_at"]
        .notna()
        .astype(float)
        .to_numpy()
    )
    base["open_time"] = open_times.to_numpy()
    return base.reset_index(drop=True)


def sample_weights(
    frame: pd.DataFrame,
    settings: Settings,
) -> np.ndarray:
    cfg = settings.section("features")
    event_weight = float(
        cfg.get("event_sample_weight", 4.0)
    )
    normal_weight = float(
        cfg.get("normal_sample_weight", 0.70)
    )
    is_event = frame["is_event"].to_numpy(dtype=bool)
    score = pd.to_numeric(
        frame.get("event_score", 0.0),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    weights = np.where(
        is_event,
        event_weight
        * (0.75 + np.clip(score, 0.0, 1.0)),
        normal_weight,
    ).astype(float)
    recency = np.linspace(
        float(
            cfg.get("oldest_sample_weight", 0.55)
        ),
        1.0,
        len(frame),
    )
    return weights * recency


def _kama(
    series: pd.Series,
    er_period: int,
    fast_period: int,
    slow_period: int,
) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    change = series.diff(er_period).abs()
    volatility = series.diff().abs().rolling(
        er_period
    ).sum()
    efficiency = (
        change / volatility.replace(0, np.nan)
    ).clip(0, 1)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    smoothing = (
        efficiency * (fast_sc - slow_sc) + slow_sc
    ) ** 2
    first = er_period
    if len(values) <= first:
        return pd.Series(result, index=series.index)
    result[first] = float(
        np.nanmean(values[: first + 1])
    )
    for index in range(first + 1, len(values)):
        if (
            not np.isfinite(values[index])
            or not np.isfinite(result[index - 1])
        ):
            continue
        coefficient = (
            float(smoothing.iloc[index])
            if np.isfinite(smoothing.iloc[index])
            else slow_sc**2
        )
        result[index] = result[index - 1] + coefficient * (
            values[index] - result[index - 1]
        )
    return pd.Series(result, index=series.index)


def _rolling_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(
        ddof=0
    ).replace(0, np.nan)
    return (series - mean) / std


def _last_percentile_rank(
    values: np.ndarray,
) -> float:
    if len(values) == 0 or not np.isfinite(values[-1]):
        return np.nan
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid <= values[-1]))
