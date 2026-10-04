from pathlib import Path

import pytest

from rag.config import ConfigError, load_settings


def test_missing_api_key_raises_helpful_error():
    with pytest.raises(ConfigError, match="NVIDIA_API_KEY"):
        load_settings({})


def test_blank_api_key_counts_as_missing():
    with pytest.raises(ConfigError):
        load_settings({"NVIDIA_API_KEY": "   "})


def test_defaults_and_derived_paths():
    s = load_settings({"NVIDIA_API_KEY": "k"})
    assert s.api_key == "k"
    assert s.chunk_strategy == "heading"
    assert s.pages_dir == Path("data/pages")
    assert s.ocr_dir == Path("data/ocr")
    assert s.chroma_dir == Path("data/chroma")


def test_env_overrides():
    s = load_settings({
        "NVIDIA_API_KEY": "k",
        "CHAT_MODEL": "x/y",
        "CHUNK_STRATEGY": "page",
        "RERANK_THRESHOLD": "0.5",
        "RETRIEVE_K": "5",
    })
    assert (s.chat_model, s.chunk_strategy, s.rerank_threshold, s.retrieve_k) == ("x/y", "page", 0.5, 5)


def test_reasoning_effort_override_and_disable():
    assert load_settings({"NVIDIA_API_KEY": "k"}).chat_reasoning_effort == "low"
    assert load_settings({"NVIDIA_API_KEY": "k", "CHAT_REASONING_EFFORT": "medium"}).chat_reasoning_effort == "medium"
    assert load_settings({"NVIDIA_API_KEY": "k", "CHAT_REASONING_EFFORT": "off"}).chat_reasoning_effort == ""


def test_ocr_workers_default_and_override():
    assert load_settings({"NVIDIA_API_KEY": "k"}).ocr_workers == 4
    assert load_settings({"NVIDIA_API_KEY": "k", "OCR_WORKERS": "2"}).ocr_workers == 2


def test_vision_provider_defaults_to_main_provider():
    s = load_settings({"NVIDIA_API_KEY": "k"})
    assert (s.vision_base_url, s.vision_api_key) == ("", "")


def test_vision_provider_override():
    s = load_settings({
        "NVIDIA_API_KEY": "k",
        "VISION_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "VISION_API_KEY": "g",
        "VISION_MODEL": "gemini-x",
    })
    assert (s.vision_base_url, s.vision_api_key, s.vision_model) == (
        "https://generativelanguage.googleapis.com/v1beta/openai/", "g", "gemini-x"
    )


def test_ingest_min_interval_override():
    assert load_settings({"NVIDIA_API_KEY": "k"}).ingest_min_interval == 1.5
    assert load_settings({"NVIDIA_API_KEY": "k", "INGEST_MIN_INTERVAL": "13"}).ingest_min_interval == 13.0


def test_repr_hides_api_keys():
    s = load_settings({"NVIDIA_API_KEY": "nvapi-secret", "VISION_API_KEY": "gemini-secret"})
    assert "nvapi-secret" not in repr(s)
    assert "gemini-secret" not in repr(s)


def test_invalid_values_raise():
    with pytest.raises(ConfigError, match="CHUNK_STRATEGY"):
        load_settings({"NVIDIA_API_KEY": "k", "CHUNK_STRATEGY": "bogus"})
    with pytest.raises(ConfigError, match="RETRIEVE_K"):
        load_settings({"NVIDIA_API_KEY": "k", "RETRIEVE_K": "many"})
