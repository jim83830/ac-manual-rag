"""所有模型 API 呼叫都集中在這裡。要換供應商（Gemini、Claude…）只需要改這個檔案。"""
from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from typing import Literal

import httpx
from openai import OpenAI, OpenAIError

from rag.config import Settings

EMBED_BATCH = 32
_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """呼叫模型 API 失敗（已重試仍失敗，或是不可重試的錯誤）。"""


class NvidiaClient:
    def __init__(
        self,
        settings: Settings,
        *,
        openai_client=None,
        http_client: httpx.Client | None = None,
        min_interval: float = 0.0,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        self._s = settings
        # openai SDK 內建 429 / 5xx 的指數退避重試
        self._openai = openai_client or OpenAI(
            base_url=settings.base_url, api_key=settings.api_key, max_retries=5, timeout=120
        )
        self._http = http_client or httpx.Client(timeout=60)
        self._min_interval = min_interval
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_call: float | None = None

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        now = self._monotonic()
        if self._last_call is not None:
            wait = self._last_call + self._min_interval - now
            if wait > 0:
                self._sleep(wait)
                now += wait
        self._last_call = now

    def ocr_page(self, image_jpeg: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_jpeg).decode("ascii")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        return self._complete(
            self._s.vision_model, messages, max_tokens=4096, temperature=0.0, timeout=self._s.ocr_timeout
        )

    def chat(self, messages: list[dict], max_tokens: int = 1024) -> str:
        return self._complete(
            self._s.chat_model, messages, max_tokens=max_tokens, temperature=0.2, extra_body=self._chat_extra_body()
        )

    def _chat_extra_body(self) -> dict | None:
        effort = self._s.chat_reasoning_effort
        return {"reasoning_effort": effort} if effort else None

    def chat_stream(self, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        self._throttle()
        try:
            stream = self._openai.chat.completions.create(
                model=self._s.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                stream=True,
                extra_body=self._chat_extra_body(),
            )
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except OpenAIError as exc:
            raise LLMError(f"聊天模型呼叫失敗：{exc}") from exc

    def _complete(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        extra_body: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        self._throttle()
        options = {"timeout": timeout} if timeout else {}
        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
                **options,
            )
        except OpenAIError as exc:
            raise LLMError(f"模型 {model} 呼叫失敗：{exc}") from exc
        return response.choices[0].message.content or ""

    def embed(self, texts: list[str], input_type: Literal["passage", "query"]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            self._throttle()
            try:
                response = self._openai.embeddings.create(
                    model=self._s.embed_model,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "END"},
                )
            except OpenAIError as exc:
                raise LLMError(f"Embedding 呼叫失敗：{exc}") from exc
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        if not passages:
            return []
        payload = {
            "model": self._s.rerank_model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            "truncate": "END",
        }
        data = self._post_json(self._s.rerank_url, payload)
        ranks = [(int(r["index"]), float(r["logit"])) for r in data["rankings"]]
        return sorted(ranks, key=lambda r: r[1], reverse=True)

    def list_models(self) -> list[str]:
        try:
            return sorted(model.id for model in self._openai.models.list())
        except OpenAIError as exc:
            raise LLMError(f"無法列出模型：{exc}") from exc

    def _post_json(self, url: str, payload: dict, attempts: int = 5) -> dict:
        headers = {"Authorization": f"Bearer {self._s.api_key}", "Accept": "application/json"}
        error = ""
        for attempt in range(attempts):
            self._throttle()
            try:
                response = self._http.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                error = f"連線失敗：{exc}"
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in _RETRY_STATUS:
                    raise LLMError(f"HTTP {response.status_code}：{response.text[:200]}")
                error = f"HTTP {response.status_code}"
            if attempt < attempts - 1:
                self._sleep(2 ** attempt)
        raise LLMError(f"重試 {attempts} 次仍失敗（{error}）")
