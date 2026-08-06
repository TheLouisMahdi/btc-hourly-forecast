"""Render the complete GitHub Pages dashboard in one deterministic pass."""

from __future__ import annotations

import github_pages_dashboard
import github_resilience_dashboard
import github_uncertainty_dashboard
import github_visual_dashboard


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
