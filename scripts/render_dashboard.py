"""Render the complete GitHub Pages dashboard in one deterministic pass."""

from __future__ import annotations

import re
from pathlib import Path

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
    ):
        status = component.main()
        if status != 0:
            return status

    _ensure_resilience_panel()
    return 0


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
