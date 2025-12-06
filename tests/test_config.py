import pytest

from s3_vectors_mcp.config import MissingSettingError, settings_from_env_with_overrides


def _seed_env(monkeypatch):
    monkeypatch.setenv("S3VECTORS_BUCKET_NAME", "bucket")
    monkeypatch.setenv("S3VECTORS_INDEX_NAME", "index")
    monkeypatch.setenv("S3VECTORS_MODEL_ID", "model")
    monkeypatch.setenv("S3VECTORS_REGION", "us-east-1")


def test_settings_override_precedence(monkeypatch):
    _seed_env(monkeypatch)
    settings = settings_from_env_with_overrides(
        model_id="override-model",
        dimensions=2048,
    )
    assert settings.model_id == "override-model"
    assert settings.dimensions == 2048


def test_missing_required_settings_raise(monkeypatch):
    monkeypatch.delenv("S3VECTORS_BUCKET_NAME", raising=False)
    monkeypatch.delenv("S3VECTORS_INDEX_NAME", raising=False)
    monkeypatch.delenv("S3VECTORS_MODEL_ID", raising=False)

    with pytest.raises(MissingSettingError):
        settings_from_env_with_overrides()

