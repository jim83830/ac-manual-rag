"""集中所有可調整的設定。模型名稱與調參數值都可用環境變數（.env）覆寫，不必改程式。"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CHUNK_STRATEGIES = ("heading", "page", "fixed")

# 介面上的「常見問題」按鈕
EXAMPLE_QUESTIONS = (
    "強力運轉怎麼開？",
    "濾網要怎麼清潔？",
    "怎麼設定定時關機？",
    "遙控器按了都沒反應",
)


class ConfigError(Exception):
    """設定有誤（缺 API key、數值格式錯誤、登記表錯誤等）。"""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    vision_model: str = "google/gemma-4-31b-it"
    chat_model: str = "z-ai/glm-5.3-flash"
    # 推理型聊天模型的思考量；"low" 讓它少想一點、快一點。空字串＝不送這個參數
    chat_reasoning_effort: str = "low"
    embed_model: str = "nvidia/nemotron-3-embed-1b"
    rerank_model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    rerank_url: str = (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"
    )
    manuals_dir: Path = Path("manuals")
    data_dir: Path = Path("data")
    render_dpi: int = 150
    ocr_max_side: int = 1600
    ocr_jpeg_quality: int = 85
    ocr_timeout: float = 360.0  # 視覺模型一頁可能要 3～4 分鐘
    chunk_strategy: str = "heading"
    chunk_max_chars: int = 800
    retrieve_k: int = 10
    rerank_k: int = 3
    rerank_threshold: float = -4.0  # 暫定：此 reranker 分數偏負，Task 13 用 eval 精調
    ingest_min_interval: float = 1.5  # 免費額度約 40 次/分鐘

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def ocr_dir(self) -> Path:
        return self.data_dir / "ocr"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


_STR_ENV = {
    "vision_model": "VISION_MODEL",
    "chat_model": "CHAT_MODEL",
    "embed_model": "EMBED_MODEL",
    "rerank_model": "RERANK_MODEL",
    "rerank_url": "RERANK_URL",
    "chunk_strategy": "CHUNK_STRATEGY",
}
_INT_ENV = {"chunk_max_chars": "CHUNK_MAX_CHARS", "retrieve_k": "RETRIEVE_K", "rerank_k": "RERANK_K"}
_FLOAT_ENV = {"rerank_threshold": "RERANK_THRESHOLD"}


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    api_key = env.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "找不到 NVIDIA_API_KEY：請把 .env.example 複製成 .env，填入 build.nvidia.com 取得的 API key"
        )
    values: dict[str, object] = {"api_key": api_key}
    for field_name, var in _STR_ENV.items():
        if env.get(var, "").strip():
            values[field_name] = env[var].strip()
    for field_name, var in _INT_ENV.items():
        if env.get(var, "").strip():
            try:
                values[field_name] = int(env[var])
            except ValueError:
                raise ConfigError(f"{var} 必須是整數，目前是 {env[var]!r}") from None
    for field_name, var in _FLOAT_ENV.items():
        if env.get(var, "").strip():
            try:
                values[field_name] = float(env[var])
            except ValueError:
                raise ConfigError(f"{var} 必須是數字，目前是 {env[var]!r}") from None
    effort = env.get("CHAT_REASONING_EFFORT", "").strip()
    if effort:
        values["chat_reasoning_effort"] = "" if effort.lower() == "off" else effort
    settings = Settings(**values)
    if settings.chunk_strategy not in CHUNK_STRATEGIES:
        raise ConfigError(
            f"CHUNK_STRATEGY 必須是 {'/'.join(CHUNK_STRATEGIES)} 其中之一，目前是 {settings.chunk_strategy!r}"
        )
    return settings
