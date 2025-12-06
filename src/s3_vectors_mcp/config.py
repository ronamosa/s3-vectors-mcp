"""
Configuration helpers for the S3 Vectors MCP server.

Settings precedence:
1. Tool/CLI overrides supplied at runtime
2. Environment variables
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Mapping, Optional


class MissingSettingError(ValueError):
    """Raised when a required configuration value is missing."""


def _require(env: Mapping[str, str], key: str) -> str:
    value = env.get(key)
    if not value:
        raise MissingSettingError(
            f"{key} must be provided via environment variable or runtime override"
        )
    return value


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


_ENV_KEYS = {
    "bucket_name": "S3VECTORS_BUCKET_NAME",
    "index_name": "S3VECTORS_INDEX_NAME",
    "model_id": "S3VECTORS_MODEL_ID",
    "region": "S3VECTORS_REGION",
    "profile": "S3VECTORS_PROFILE",
    "dimensions": "S3VECTORS_DIMENSIONS",
    "log_level": "S3VECTORS_LOG_LEVEL",
}


@dataclass(frozen=True)
class Settings:
    bucket_name: str
    index_name: str
    model_id: str
    region: Optional[str] = None
    profile: Optional[str] = None
    dimensions: Optional[int] = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = env or os.environ
        return cls(
            bucket_name=_require(env, "S3VECTORS_BUCKET_NAME"),
            index_name=_require(env, "S3VECTORS_INDEX_NAME"),
            model_id=_require(env, "S3VECTORS_MODEL_ID"),
            region=env.get("S3VECTORS_REGION"),
            profile=env.get("S3VECTORS_PROFILE"),
            dimensions=_optional_int(env.get("S3VECTORS_DIMENSIONS")),
            log_level=env.get("S3VECTORS_LOG_LEVEL", "INFO"),
        )

    def with_overrides(self, **overrides) -> "Settings":
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        if "dimensions" in clean_overrides and isinstance(
            clean_overrides["dimensions"], str
        ):
            clean_overrides["dimensions"] = int(clean_overrides["dimensions"])
        return replace(self, **clean_overrides)


def settings_from_env_with_overrides(**overrides) -> Settings:
    env = dict(os.environ)
    for field, value in overrides.items():
        if value is None:
            continue
        env_key = _ENV_KEYS.get(field)
        if not env_key:
            continue
        env[env_key] = str(value)
    return Settings.from_env(env)

