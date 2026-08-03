from __future__ import annotations

import html
import json
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import Settings
from .features import build_feature_set
from .market import MarketDataClient
from .model import latest_bundle
from .runtime import RuntimeEngine, start_background_engine
from .storage import Database

CSS = r"""
:root{--bg:#0b0e11;--surface:#15191f;--surface2:#1b2028;--border:#2a3039;--text:#f5f7fa;--muted:#9aa4b2;--green:#16c784;--red:#ea3943;--amber:#f5b73b}
.gradio-container{max-width:1500px!important;margin:0 auto!important;background:var(--bg)!important;color:var(--text)!important}.dark,body{background:var(--bg)!important}
.hero{display:flex;justify-content:space-between;gap:16px;align-items:center;padding:19px 22px;border:1px solid var(--border);border-radius:16px;background:linear-gradient(135deg,#171c24,#11151b);margin-bottom:12px}.brand{display:flex;gap:13px;align-items:center}.coin{width:46px;height:46px;border-radius:50%;background:#f7931a;display:flex;align-items:center;justify-content:center;font-size:27px;font-weight:900}.title{font-size:22px;font-weight:800}.sub{font-size:12px;color:var(--muted);margin-top:5px}.right{text-align:right;font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:12px 0}.card{padding:14px;border:1px solid var(--border);border-radius:13px;background:var(--surface);min-height:91px}.label{font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase}.value{font-size:21px;font-weight:800;margin-top:8px}.meta{font-size:11px;color:var(--muted);margin-top:7px;line-height:1.35}.up{color:var(--green)}.down{color:var(--red)}.wait{color:var(--amber)}.banner{padding:13px 15px;border-radius:12px;border:1px solid var(--border);font-weight:750;margin:10px 0}.good{color:var(--green);border-color:rgba(22,199,132,.5)}.bad{color:var(--red);border-color:rgba(234,57,67,.5)}.warn{color:var(--amber);border-color:rgba(245,183,59,.5)}footer{display:none!important}
@media(max-width:1100px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:650px){.grid{grid-template-columns:repeat(2,1fr)}.hero{align-items:flex-start;flex-direction:column}.right{text-align:left}}
"""


class Dashboard:
    def __init__(self, settings: Settings, database: Database, start_engine: bool = True):
        self.settings = settings
        self.database = database
        self.market = MarketDataClient(settings)
        self.engine = start_background_engine(settings, database) if start_engine else RuntimeEngine(settings, database)

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="BTC Hourly Regime Trader") as app:
            header = gr.HTML()
            cards = gr.HTML()
            status_banner = gr.HTML()
            with gr.Row():
                refresh_button = gr.Button("Refresh", variant="primary")
                reset_clock = gr.Button("Start suggestions from the next candle")
            reset_message = gr.Markdown()

            with gr.Tabs():
                with gr.Tab("Overview"):
                    market_chart = gr.Plot(label="BTC 1h + adaptive regime structure")
                    current_decision = gr.Dataframe(label="Current decision", interactive=False, wrap=True)
                with gr.Tab("Forecast 1–3h"):
                    probability_chart = gr.Plot(label="UP/DOWN and tradeability probabilities")
                    forecast_table = gr.Dataframe(label="Horizon forecasts", interactive=False, wrap=True)
                    target_tracking = gr.Plot(label="Resolved forecast tracking")
                    event_study = gr.Plot(label="Historical return after independent market events")
                with gr.Tab("Trade plan"):
                    trade_plan = gr.Dataframe(label="Risk-controlled paper trade plan", interactive=False, wrap=True)
                    blockers = gr.Dataframe(label="Fail-safe checks", interactive=False, wrap=True)
                with gr.Tab("Model quality"):
                    qualification = gr.HTML()
                    metrics = gr.Dataframe(label="Walk-forward metrics", interactive=False, wrap=True)
                with gr.Tab("News"):
                    news_summary = gr.Dataframe(label="Hourly news features", interactive=False, wrap=True)
                    news_items = gr.Dataframe(label="Recent BTC headlines", interactive=False, wrap=True)
                with gr.Tab("Signal history"):
                    equity_chart = gr.Plot(label="Resolved paper-signal equity curve")
                    signal_history = gr.Dataframe(label="Signals", interactive=False, wrap=True)
                with gr.Tab("System"):
                    system_status = gr.JSON(label="Runtime and fail-safe status")
                    events = gr.Dataframe(label="Recent runtime events", interactive=False, wrap=True)

            outputs = [
                header, cards, status_banner, market_chart, current_decision, probability_chart,
                forecast_table, target_tracking, event_study, trade_plan, blockers, qualification, metrics,
                news_summary, news_items, equity_chart, signal_history, system_status, events,
            ]
            refresh_button.click(self.refresh, outputs=outputs)
            reset_clock.click(self.reset_session, outputs=[reset_message, system_status])
            timer = gr.Timer(value=10)
            timer.tick(self.refresh, outputs=outputs)
            app.load(self.refresh, outputs=outputs)
        return app

    def reset_session(self):
        state = self.engine.reset_session_clock()
        return "Suggestions begin only after the next fully closed one-hour candle.", state

    def refresh(self):
        runtime = self.engine.status()
        try:
            bundle = latest_bundle(self.settings)
        except Exception:
            bundle = None
        provider = bundle.provider if bundle else self._best_provider()
        candles = self.database.load_candles(
            provider=provider,
            symbol=self.settings.section("market").get("symbol", "BTCUSDT"),
            limit=500,
        ) if provider else pd.DataFrame()
        news = self.database.load_news(start=candles["open_time"].min() if not candles.empty else None)
        feature_frame = pd.DataFrame()
        if not candles.empty:
            try:
                feature_frame = build_feature_set(candles, news, self.settings, include_labels=True).frame
            except Exception:
                feature_frame = pd.DataFrame()
        latest_signal_df = self.database.recent_signals(limit=1)
        latest_signal = latest_signal_df.iloc[0].to_dict() if not latest_signal_df.empty else None
        try:
            quote = self.market.live_quote(provider_hint=provider)
        except Exception:
            quote = None
        all_signals = self.database.recent_signals(limit=100)
        probability_chart, forecast_table = self._forecast_outputs(latest_signal)
        trade_plan, blockers = self._trade_outputs(latest_signal)
        return (
            self._header(provider, bundle, runtime),
            self._cards(quote, feature_frame, latest_signal, runtime),
            self._status_banner(bundle, latest_signal, runtime),
            self._market_chart(feature_frame, all_signals),
            self._decision_table(latest_signal),
            probability_chart,
            forecast_table,
            self._target_tracking_chart(all_signals),
            self._event_study(feature_frame),
            trade_plan,
            blockers,
            self._qualification(bundle),
            self._metrics(bundle),
            self._news_summary(feature_frame),
            self._news_items(news),
            self._equity_chart(all_signals),
            self._signal_history(all_signals),
            self._system_status(provider, candles, news, bundle, runtime, quote, feature_frame),
            self.database.recent_events(limit=50),
        )

    def _best_provider(self) -> str | None:
        providers = self.database.providers(self.settings.section("market").get("symbol", "BTCUSDT"))
        return str(providers[0]["provider"]) if providers else None

    def _header(self, provider: str | None, bundle: Any, runtime: dict[str, Any]) -> str:
        model_text = bundle.model_id if bundle else "No v2 model"
        return f"""
        <div class='hero'><div class='brand'><div class='coin'>₿</div><div><div class='title'>BTC Hourly Regime Trader</div>
        <div class='sub'>KAMA regime + ADX • Donchian breakout • squeeze release • pullback resume • volume impulse • 1h candles</div></div></div>
        <div class='right'><b>{html.escape(provider or 'No market data')}</b><br>{html.escape(model_text)}<br>Runtime: {html.escape(str(runtime.get('status')))}</div></div>
        """

    def _cards(self, quote: Any, features: pd.DataFrame, signal: dict[str, Any] | None, runtime: dict[str, Any]) -> str:
        latest = features.iloc[-1] if not features.empty else None
        price = quote.price if quote else (float(latest["close"]) if latest is not None else np.nan)
        kama = float(latest["kama"]) if latest is not None and pd.notna(latest.get("kama")) else np.nan
        adx = float(latest["adx"]) if latest is not None and pd.notna(latest.get("adx")) else np.nan
        regime = str(latest.get("regime", "—")) if latest is not None else "—"
        event = str(latest.get("event_type", "NONE")) if latest is not None else "NONE"
        direction = str(signal.get("forecast_direction")) if signal else "Waiting"
        action = str(signal.get("action")) if signal else "WAIT"
        confidence = float(signal.get("confidence", 0)) if signal else 0
        seconds = int(runtime.get("seconds_to_first_eligible", 0))
        return "<div class='grid'>" + "".join([
            self._card("Live BTC price", _money(price), quote.provider if quote else "closed candle fallback"),
            self._card("Adaptive trend", _money(kama), f"Price distance: {_bps(price/kama-1 if np.isfinite(price) and np.isfinite(kama) else np.nan)}"),
            self._card("ADX strength", f"{adx:.1f}" if np.isfinite(adx) else "—", regime),
            self._card("New event", event, f"Score: {_fmt(latest.get('event_score')) if latest is not None else '—'}", "up" if latest is not None and int(latest.get("event_direction",0))>0 else "down" if latest is not None and int(latest.get("event_direction",0))<0 else "wait"),
            self._card("Model forecast", direction, f"Confidence {confidence:.1%}", "up" if direction == "UP" else "down" if direction == "DOWN" else "wait"),
            self._card("Trade action", action, f"Next-candle gate: {seconds//60}m {seconds%60}s", "up" if action == "LONG" else "down" if action == "SHORT" else "wait"),
        ]) + "</div>"

    def _card(self, label: str, value: str, meta: str, cls: str = "") -> str:
        return f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value {cls}'>{html.escape(value)}</div><div class='meta'>{html.escape(meta)}</div></div>"

    def _status_banner(self, bundle: Any, signal: dict[str, Any] | None, runtime: dict[str, Any]) -> str:
        if bundle is None:
            return "<div class='banner bad'>No compatible v2 model. Run start_retrain.bat.</div>"
        if runtime.get("status") == "FAIL_SAFE":
            return f"<div class='banner bad'>FAIL-SAFE: {html.escape(str(runtime.get('last_error')))}</div>"
        if not bundle.qualification.get("passed", False):
            return "<div class='banner warn'>UP/DOWN forecasts are visible, but trade actions remain blocked until at least one horizon passes event-based walk-forward qualification.</div>"
        if signal and signal.get("action") in {"LONG", "SHORT"}:
            return f"<div class='banner good'>Qualified paper suggestion: {html.escape(str(signal['action']))}</div>"
        return "<div class='banner warn'>Forecasting is active; no qualified new event is actionable now.</div>"

    def _market_chart(self, frame: pd.DataFrame, signals: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if frame.empty:
            return fig.update_layout(template="plotly_dark", title="No candle data")
        view = frame.tail(240)
        fig.add_trace(go.Candlestick(x=view["open_time"], open=view["open"], high=view["high"], low=view["low"], close=view["close"], name="BTC"))
        fig.add_trace(go.Scatter(x=view["open_time"], y=view["kama"], mode="lines", name="KAMA adaptive trend"))
        fig.add_trace(go.Scatter(x=view["open_time"], y=view["donchian_high"], mode="lines", name="Donchian high", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=view["open_time"], y=view["donchian_low"], mode="lines", name="Donchian low", line=dict(dash="dot")))
        events = view[view["is_event"] == 1]
        if not events.empty:
            fig.add_trace(go.Scatter(x=events["open_time"], y=events["close"], mode="markers", marker_symbol="diamond", marker_size=10, text=events["event_type"], name="Independent event"))
        if not signals.empty:
            markers = signals[signals["action"].isin(["LONG", "SHORT"])]
            if not markers.empty:
                fig.add_trace(go.Scatter(x=markers["candle_time"], y=markers["price"], mode="markers", marker_symbol="triangle-up", marker_size=13, name="Paper suggestion"))
        fig.update_layout(template="plotly_dark", height=560, margin=dict(l=30,r=20,t=35,b=30), xaxis_rangeslider_visible=False, legend_orientation="h")
        return fig

    def _decision_table(self, signal: dict[str, Any] | None) -> pd.DataFrame:
        if signal is None:
            return pd.DataFrame([{"Status": "Waiting for the first eligible closed candle"}])
        return pd.DataFrame([{
            "Candle": signal.get("candle_time"), "Event": signal.get("event_type"), "Regime": signal.get("regime"),
            "Forecast": signal.get("forecast_direction"), "Action": signal.get("action"),
            "P(event continuation)": f"{float(signal.get('confidence',0)):.2%}",
            "Tradeability": f"{float(signal.get('tradeability_probability') or 0):.2%}",
            "Expected return": f"{float(signal.get('expected_return',0)):.3%}",
            "Stress net edge": f"{float(signal.get('expected_net_edge_bps',0)):.1f} bps", "Horizon": f"{signal.get('selected_horizon')}h",
        }])

    def _forecast_outputs(self, signal: dict[str, Any] | None):
        fig = go.Figure()
        if signal is None:
            return fig.update_layout(template="plotly_dark"), pd.DataFrame([{"Status": "No signal yet"}])
        probabilities = _json(signal.get("probabilities_json"))
        tradeability = _json(signal.get("tradeability_json"))
        returns = _json(signal.get("returns_json"))
        horizons = sorted(int(k) for k in probabilities)
        up = [float(probabilities.get(str(h), probabilities.get(h, .5))) for h in horizons]
        trade = [float(tradeability.get(str(h), tradeability.get(h, .5))) for h in horizons]
        fig.add_trace(go.Bar(x=[f"{h}h" for h in horizons], y=up, name="P(UP)"))
        fig.add_trace(go.Bar(x=[f"{h}h" for h in horizons], y=[1-x for x in up], name="P(DOWN)"))
        fig.add_trace(go.Scatter(x=[f"{h}h" for h in horizons], y=trade, mode="lines+markers", name="P(tradeable)"))
        fig.update_layout(template="plotly_dark", barmode="group", yaxis_range=[0,1], height=380)
        rows = []
        for h, p, t in zip(horizons, up, trade):
            r = float(returns.get(str(h), returns.get(h, 0)))
            rows.append({"Horizon":f"{h}h","Direction":"UP" if p>=.5 else "DOWN","P(UP)":p,"P(DOWN)":1-p,"P(tradeable)":t,"Expected return":r,"Predicted target":float(signal.get("price",0))*(1+r)})
        return fig, pd.DataFrame(rows)

    def _target_tracking_chart(self, signals: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if signals.empty:
            return fig.update_layout(template="plotly_dark", title="No resolved signals yet")
        resolved = signals[signals["resolved"] == 1].sort_values("candle_time")
        if resolved.empty:
            return fig.update_layout(template="plotly_dark", title="No resolved signals yet")
        fig.add_trace(go.Scatter(x=resolved["candle_time"], y=resolved["realized_return"].astype(float)*100, mode="lines+markers", name="Net realized return %"))
        fig.add_hline(y=0)
        fig.update_layout(template="plotly_dark", height=420, yaxis_title="Net return (%)")
        return fig

    def _event_study(self, frame: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if frame.empty or "future_return_h1" not in frame:
            return fig.update_layout(template="plotly_dark")
        events = frame[frame["is_event"] == 1].copy()
        if events.empty:
            return fig.update_layout(template="plotly_dark", title="No events in visible history")
        for event_type, subset in events.groupby("event_type"):
            values = [subset[f"future_return_h{h}"].mean()*100 for h in (1,2,3)]
            fig.add_trace(go.Bar(x=["1h","2h","3h"], y=values, name=str(event_type)))
        fig.update_layout(template="plotly_dark", barmode="group", height=420, yaxis_title="Mean raw return (%)")
        return fig

    def _trade_outputs(self, signal: dict[str, Any] | None):
        if signal is None:
            return pd.DataFrame([{"Status":"No trade plan yet"}]), pd.DataFrame([{"Check":"Waiting","Result":"BLOCK"}])
        plan = _json(signal.get("trade_plan_json")); blockers = _json(signal.get("blockers_json"))
        plan_df = pd.DataFrame([{"Field":k,"Value":v} for k,v in plan.items()])
        blocker_list = blockers if isinstance(blockers,list) else []
        blocker_df = pd.DataFrame([{"Check":b,"Result":"BLOCK"} for b in blocker_list] or [{"Check":"All configured gates","Result":"PASS"}])
        return plan_df, blocker_df

    def _qualification(self, bundle: Any) -> str:
        if bundle is None:
            return "<div class='banner bad'>No compatible model.</div>"
        q=bundle.qualification; cls="good" if q.get("passed") else "bad"; text="PASSED" if q.get("passed") else "BLOCKED"
        horizons=", ".join(f"{h}h" for h in q.get("qualified_horizons",[])) or "none"
        blockers="<br>".join(html.escape(x) for x in q.get("blockers",[]))
        return f"<div class='banner {cls}'>Model qualification: {text} • Qualified horizons: {horizons}<div class='meta'>{blockers}</div></div>"

    def _metrics(self, bundle: Any) -> pd.DataFrame:
        if bundle is None:return pd.DataFrame()
        rows=[]
        for horizon,m in bundle.metrics.items():
            rows.append({"Horizon":f"{horizon}h","Samples":m.get("samples"),"AUC":m.get("auc"),"Balanced accuracy":m.get("balanced_accuracy"),"Generic calibration error":m.get("calibration_error"),"Continuation calibration error":m.get("event_calibration_error"),"Independent events":m.get("event_samples"),"Continuation AUC":m.get("event_auc"),"Tradeability AUC":m.get("tradeability_auc"),"OOF trades":m.get("trading",{}).get("selected"),"Mean net return":m.get("trading",{}).get("mean_net_return"),"Positive folds":m.get("positive_fold_fraction")})
        return pd.DataFrame(rows)

    def _news_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:return pd.DataFrame()
        row=frame.iloc[-1]; cols=[c for c in frame.columns if c.startswith("news_")]
        return pd.DataFrame([{"Feature":c,"Value":row[c]} for c in cols])

    def _news_items(self, news: pd.DataFrame) -> pd.DataFrame:
        if news.empty:return pd.DataFrame([{"Status":"No news collected"}])
        return news.tail(50)[["published_at","first_seen_at","source","title","sentiment","relevance"]].sort_values("published_at",ascending=False)

    def _equity_chart(self, signals: pd.DataFrame) -> go.Figure:
        fig=go.Figure()
        if signals.empty:return fig.update_layout(template="plotly_dark")
        resolved=signals[(signals["resolved"]==1)&signals["realized_return"].notna()].sort_values("candle_time").copy()
        if resolved.empty:return fig.update_layout(template="plotly_dark",title="No resolved paper trades yet")
        resolved["equity"]=float(self.settings.section("strategy").get("account_equity_usd",1000))*(1+resolved["realized_return"].astype(float)).cumprod()
        fig.add_trace(go.Scatter(x=resolved["candle_time"],y=resolved["equity"],mode="lines+markers",name="Paper equity"))
        fig.update_layout(template="plotly_dark",height=390)
        return fig

    def _signal_history(self, signals: pd.DataFrame) -> pd.DataFrame:
        if signals.empty:return pd.DataFrame()
        cols=["candle_time","event_type","regime","forecast_direction","action","confidence","tradeability_probability","expected_net_edge_bps","selected_horizon","qualification_passed","resolved","entry_price","exit_price","realized_return","outcome"]
        return signals[[c for c in cols if c in signals]].copy()

    def _system_status(self, provider: str|None, candles: pd.DataFrame, news: pd.DataFrame, bundle: Any, runtime: dict[str,Any], quote: Any, features: pd.DataFrame) -> dict[str,Any]:
        gaps=[]
        if not candles.empty:gaps=pd.DatetimeIndex(candles["open_time"]).to_series().diff().dropna().dt.total_seconds().div(3600).tolist()
        latest=features.iloc[-1] if not features.empty else None
        return {"runtime":runtime,"provider":provider,"candle_rows":len(candles),"last_closed_candle":None if candles.empty else candles["open_time"].max().isoformat(),"maximum_gap_hours":max(gaps) if gaps else 0,"news_rows":len(news),"last_news":None if news.empty else news["published_at"].max().isoformat(),"model_id":None if bundle is None else bundle.model_id,"model_created_at":None if bundle is None else bundle.created_at,"qualification":None if bundle is None else bundle.qualification,"current_regime":None if latest is None else latest.get("regime"),"current_event":None if latest is None else latest.get("event_type"),"live_quote":None if quote is None else {"provider":quote.provider,"price":quote.price,"timestamp":quote.timestamp.isoformat()},"paper_only":True,"live_order_execution_implemented":False,"model_schema_version":None if bundle is None else getattr(bundle,"schema_version",1)}


def launch_dashboard(settings: Settings, database: Database, share: bool = False, start_engine: bool = True) -> None:
    app=Dashboard(settings,database,start_engine=start_engine).build(); live_cfg=settings.section("live")
    app.launch(server_name=str(live_cfg.get("dashboard_host","127.0.0.1")),server_port=int(live_cfg.get("dashboard_port",7860)),share=share,inbrowser=True,theme=gr.themes.Base(),css=CSS)


def _json(value: Any) -> Any:
    if isinstance(value,(dict,list)):return value
    try:return json.loads(value or "{}")
    except Exception:return {}
def _money(value: float)->str:return "—" if not np.isfinite(value) else f"${value:,.2f}"
def _fmt(value: Any)->str:
    try:return f"{float(value):.2f}"
    except Exception:return "—"
def _bps(value: float)->str:return "—" if not np.isfinite(value) else f"{value*10000:+.1f} bps"
