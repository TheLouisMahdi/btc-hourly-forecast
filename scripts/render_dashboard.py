"""Render the complete GitHub Pages dashboard in one deterministic pass."""

from __future__ import annotations

import re
from pathlib import Path

import github_assistant_dashboard
import github_chart_dashboard
import github_crypto_logo_dashboard
import github_market_price_dashboard
import github_product_surface
import github_pages_dashboard
import github_resilience_dashboard
import github_uncertainty_dashboard
import github_visual_dashboard

RESILIENCE_HEADING = "Data continuity &amp; learning safety"


def main() -> int:
    """Run the base renderer followed by all presentation contracts."""
    status = github_pages_dashboard.main()
    if status != 0:
        return status

    for component in (
        github_visual_dashboard,
        github_uncertainty_dashboard,
        github_resilience_dashboard,
        github_assistant_dashboard,
        github_chart_dashboard,
        github_market_price_dashboard,
        github_crypto_logo_dashboard,
    ):
        status = component.main()
        if status != 0:
            return status

    _ensure_resilience_panel()
    status = github_product_surface.main()
    if status != 0:
        return status
    _append_historical_reassessment_link()
    return 0


def _append_historical_reassessment_link(
    index_path: Path | None = None,
    report_path: Path | None = None,
) -> None:
    """Link to a clearly labeled immutable-ledger replay report, if present."""
    root = Path(__file__).resolve().parents[1]
    index_path = index_path or root / "site" / "index.html"
    report_path = report_path or root / ".github_state" / "historical_reassessment.json"
    if not report_path.exists():
        return
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("report_type") != "HISTORICAL_REASSESSMENT_NOT_COUNTERFACTUAL_BACKTEST":
        return
    document = index_path.read_text(encoding="utf-8")
    if 'data-historical-audit="v1"' in document:
        return
    block = (
        '<p data-historical-audit="v1" class="historical-audit-note">'
        '<a href="historical_reassessment.json">Historical results audit</a>'
        ' · Original realized results preserved; alternative exit replay is '
        'research-only, not a backtest.'
        '</p>'
    )
    if "</main>" not in document:
        raise RuntimeError("No dashboard anchor for historical audit link")
    index_path.write_text(
        document.replace("</main>", block + "\n</main>", 1),
        encoding="utf-8",
    )


def _ensure_resilience_panel(index_path: Path | None = None) -> None:
    """Insert the resilience panel when a renamed ledger class hid its anchor."""
    root = Path(__file__).resolve().parents[1]
    index_path = index_path or root / "site" / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    document = index_path.read_text(encoding="utf-8")
    if RESILIENCE_HEADING in document:
        return

    site_dir = index_path.parent
    latest = github_resilience_dashboard._load_json(
        site_dir / "latest.json",
        {},
    )
    history = github_resilience_dashboard._load_json(
        site_dir / "history.json",
        [],
    )
    history = history if isinstance(history, list) else []
    panel = github_resilience_dashboard._resilience_panel(latest, history)

    ledger = re.search(
        r'<section class="[^"]*\bledger\b[^"]*">',
        document,
    )
    if ledger is not None:
        document = (
            document[: ledger.start()]
            + panel
            + "\n"
            + document[ledger.start() :]
        )
    elif "</main>" in document:
        document = document.replace("</main>", panel + "\n</main>", 1)
    else:
        raise RuntimeError("No dashboard insertion anchor is available")

    if RESILIENCE_HEADING not in document:
        raise RuntimeError("Resilience dashboard contract was not inserted")
    index_path.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
