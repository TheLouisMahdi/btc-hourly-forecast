from __future__ import annotations

from .adaptive import AdaptiveEngine as BaseAdaptiveEngine
from .config import Settings
from .model import HourlyModelBundle


class AdaptiveEngine(BaseAdaptiveEngine):
    def __init__(
        self,
        settings: Settings,
        bundle: HourlyModelBundle,
    ):
        super().__init__(settings, bundle)
        if self.state.champion_model_id != bundle.model_id:
            rebase_count = int(self.state.rebase_count) + 1
            self.state = self._new_state(
                rebase_count=rebase_count,
            )
            self.save()
