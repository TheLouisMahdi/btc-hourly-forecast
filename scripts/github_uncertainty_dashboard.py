from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

MARKER = 'data-uncertainty-display="range-only-v1"'


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    index_path = site_dir / "index.html"
    latest_path = site_dir / "latest.json"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    document = index_path.read_text(encoding="utf-8")
    latest = _load_json(latest_path, {})
    contract = latest.get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}

    document = document.replace(
        "Adaptive BTC paper positions and an exact secondary next-close forecast",
        "Adaptive BTC paper positions and a calibrated secondary next-close range",
    )
    document = re.sub(
        r'<aside class="forecast-card">.*?</aside>',
        _forecast_card(contract, latest),
        document,
        count=1,
        flags=re.DOTALL,
    )
    document = document.replace("<th>Expected close</th>", "")
    document = _remove_expected_close_cells(document)
    document = re.sub(
        r'<line[^>]*class="median"[^>]*/>',
        "",
        document,
        flags=re.DOTALL,
    )
    document = document.replace(
        "Price uncertainty is reported separately.",
        "The closing range is calibrated separately; no exact close is published.",
    )
    document = document.replace(
        "Closed-candle prices, direction outcomes and current model uncertainty",
        "Closed-candle prices, direction outcomes and the calibrated closing range",
    )
    if MARKER not in document:
        document = document.replace(
            "<body",
            f'<body {MARKER}',
            1,
        )
    index_path.write_text(document, encoding="utf-8")
    return 0


def _forecast_card(contract: dict[str, Any], latest: dict[str, Any]) -> str:
    low = _price(contract.get("likely_close_low"))
    high = _price(contract.get("likely_close_high"))
    probability = _number(contract.get("interval_probability"))
    coverage = "—" if probability is None else f"{probability * 100:.0f}% calibrated range"
    source = str(contract.get("forecast_source") or "BATCH_CHAMPION")
    target = _compact_time(contract.get("target_close_time"))
    evaluation_source = (
        latest.get("evaluation_available_at")
        if latest.get("prediction_result") == "PENDING"
        else latest.get("resolved_at")
    )
    evaluation = _compact_time(evaluation_source)
    return f'''
<aside class="forecast-card">
  <div class="label">Likely next-close range</div>
  <div class="expected">{html.escape(low)} – {html.escape(high)}</div>
  <div class="label" style="margin-top:6px">Range-only forecast · no exact close is published</div>
  <div class="range">
    <div class="range-values">
      <span>{html.escape(low)}</span>
      <strong>{html.escape(coverage)}</strong>
      <span>{html.escape(high)}</span>
    </div>
  </div>
  <div class="meta"><span>Calibration</span><strong>{html.escape(str(contract.get("interval_method") or "—").replace("_", " ").title())}</strong></div>
  <div class="meta"><span>Forecast source</span><strong>{html.escape(source.replace("_", " ").title())}</strong></div>
  <div class="meta"><span>Target closes</span><strong>{html.escape(target)}</strong></div>
  <div class="meta"><span>Evaluation</span><strong>{html.escape(evaluation)}</strong></div>
</aside>'''


def _remove_expected_close_cells(document: str) -> str:
    section_pattern = re.compile(
        r'<section class="panel ledger">.*?</section>',
        re.DOTALL,
    )

    def section_replacer(section_match: re.Match[str]) -> str:
        section = section_match.group(0)

        def row_replacer(row_match: re.Match[str]) -> str:
            row = row_match.group(0)
            cells = re.findall(r"<td>.*?</td>", row, flags=re.DOTALL)
            if len(cells) >= 9:
                return row.replace(cells[4], "", 1)
            return row

        return re.sub(
            r"<tr>.*?</tr>",
            row_replacer,
            section,
            flags=re.DOTALL,
        )

    return section_pattern.sub(section_replacer, document, count=1)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _price(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"${number:,.2f}"


def _compact_time(value: Any) -> str:
    if not value:
        return "—"
    text = str(value).replace("T", " ")
    return text[:16] + " UTC" if len(text) >= 16 else text


if __name__ == "__main__":
    raise SystemExit(main())
