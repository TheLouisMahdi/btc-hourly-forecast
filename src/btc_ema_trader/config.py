from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    root: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section {name!r} must be a mapping")
        return value

    def path(self, key: str) -> Path:
        raw = self.section("paths").get(key)
        if not raw:
            raise KeyError(f"Missing paths.{key}")
        path = Path(str(raw))
        return path if path.is_absolute() else (self.root / path).resolve()

    def ensure_runtime_dirs(self) -> None:
        self.path("database").parent.mkdir(parents=True, exist_ok=True)
        self.path("runtime_state").parent.mkdir(parents=True, exist_ok=True)
        self.path("adaptive_state").parent.mkdir(parents=True, exist_ok=True)
        for key in ("model_dir", "report_dir", "log_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    explicit = config_path or os.environ.get("BTC_EMA_CONFIG")
    if explicit:
        config_file = Path(explicit).expanduser().resolve()
        root = (
            config_file.parent.parent
            if config_file.parent.name == "config"
            else config_file.parent
        )
    else:
        root = Path.cwd().resolve()
        config_file = root / "config" / "default.yaml"
        if not config_file.exists():
            package_root = Path(__file__).resolve().parents[2]
            candidate = package_root / "config" / "default.yaml"
            if candidate.exists():
                root, config_file = package_root, candidate
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    values = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    settings = Settings(root=root, values=values)
    settings.ensure_runtime_dirs()
    return settings
