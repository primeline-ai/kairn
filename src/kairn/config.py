"""Kairn configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_BOOL_FALSY = {"false", "0", "no", "off", ""}
_BOOL_TRUTHY = {"true", "1", "yes", "on"}


def _coerce_bool(value: object) -> bool:
    """bool("false") is True - YAML string values need real parsing
    (weakness-audit rank 61)."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _BOOL_FALSY:
        return False
    if text in _BOOL_TRUTHY:
        return True
    raise ValueError(f"Cannot interpret {value!r} as a boolean config value")


@dataclass
class Config:
    """Kairn configuration."""

    workspace_path: Path = field(default_factory=lambda: Path.home() / ".kairn")
    default_workspace: str = "default"
    log_level: str = "INFO"
    pagination_default: int = 10
    pagination_max: int = 50
    response_token_limit: int = 5000
    fts5_enabled: bool = True
    wal_mode: bool = True

    # Semantic recall (opt-in): a local-embedding cosine rerank over the FTS5
    # top-N with a cosine abstain floor. Default OFF keeps the zero-dependency,
    # air-gapped, keyword-only product identity byte-for-byte. When on, node
    # writes embed via a LOCAL Ollama server (embedding_host) so corpus content
    # never leaves the machine.
    semantic_recall: bool = False
    embedding_model: str = "bge-m3"
    embedding_host: str = "http://localhost:11434"
    semantic_recall_floor: float = 0.5
    semantic_recall_top_n: int = 30

    # Decay half-lives (days). DEPRECATED as a source: the live decay path is
    # core/experience.py:HALF_LIVES (decay_rate_for_type below delegates to it).
    # Retained for backward compatibility, mirroring the calibrated 2026-06-13
    # values; do not read directly - call decay_rate_for_type() (one source).
    decay_solution: float = 120.0
    decay_pattern: float = 90.0
    decay_decision: float = 100.0
    decay_workaround: float = 40.0
    decay_gotcha: float = 70.0

    # Promotion threshold
    promotion_access_count: int = 5

    # Confidence multipliers for decay
    confidence_multiplier_high: float = 1.0
    confidence_multiplier_medium: float = 2.0
    confidence_multiplier_low: float = 4.0

    @classmethod
    def load(cls, workspace_path: Path | None = None) -> Config:
        """Load config from YAML file, env vars, then defaults."""
        config = cls()

        if workspace_path:
            config.workspace_path = workspace_path

        # Override from env
        env_path = os.environ.get("KAIRN_WORKSPACE")
        if env_path:
            config.workspace_path = Path(env_path)

        env_log = os.environ.get("KAIRN_LOG_LEVEL")
        if env_log:
            config.log_level = env_log

        # Load YAML config if exists
        config_file = config.workspace_path / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                if hasattr(config, key):
                    expected_type = type(getattr(config, key))
                    if expected_type is Path:
                        setattr(config, key, Path(value))
                    elif expected_type is bool:
                        setattr(config, key, _coerce_bool(value))
                    else:
                        setattr(config, key, expected_type(value))

        return config

    @property
    def workspaces_dir(self) -> Path:
        return self.workspace_path / "workspaces"

    @property
    def metadata_db_path(self) -> Path:
        return self.workspace_path / "metadata.db"

    def workspace_db_path(self, workspace_id: str) -> Path:
        return self.workspaces_dir / f"ws_{workspace_id}.db"

    def decay_rate_for_type(self, experience_type: str) -> float:
        """Convert half-life to decay rate: rate = ln(2) / half_life.

        Delegates to the single source of truth, core/experience.py:HALF_LIVES,
        so config and the live engine cannot drift. Local import avoids an
        import cycle (config has no module-level engine dependency).
        """
        from kairn.core.experience import HALF_LIVES, decay_rate_from_half_life

        half_life = HALF_LIVES.get(experience_type, HALF_LIVES["solution"])
        return decay_rate_from_half_life(half_life)

    def confidence_multiplier(self, confidence: str) -> float:
        """Get decay multiplier for confidence level."""
        multipliers = {
            "high": self.confidence_multiplier_high,
            "medium": self.confidence_multiplier_medium,
            "low": self.confidence_multiplier_low,
        }
        return multipliers.get(confidence, self.confidence_multiplier_high)

    def save(self) -> None:
        """Save current config to YAML."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        config_file = self.workspace_path / "config.yaml"
        data = {
            "default_workspace": self.default_workspace,
            "log_level": self.log_level,
            "pagination_default": self.pagination_default,
            "pagination_max": self.pagination_max,
            "response_token_limit": self.response_token_limit,
            "fts5_enabled": self.fts5_enabled,
            "wal_mode": self.wal_mode,
        }
        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
