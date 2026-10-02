"""測試用替身：不連網、結果可預測。"""
from __future__ import annotations

import hashlib

from rag.llm import LLMError


def fake_vector(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255 + 0.01 for b in digest[:dim]]


class FakeLLM:
    def __init__(self, *, ocr_results=(), chat_reply="", stream_parts=(), rankings=None, error_on=()):
        self.ocr_results = list(ocr_results)
        self.chat_reply = chat_reply
        self.stream_parts = list(stream_parts)
        self.rankings = rankings
        self.error_on = set(error_on)
        self.calls: list[str] = []
        self.last_messages: list[dict] | None = None

    def _record(self, name):
        self.calls.append(name)
        if name in self.error_on:
            raise LLMError(f"fake {name} failure")

    def ocr_page(self, image_jpeg, prompt):
        self._record("ocr")
        result = self.ocr_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def chat(self, messages, max_tokens=1024):
        self._record("chat")
        self.last_messages = messages
        return self.chat_reply

    def chat_stream(self, messages, max_tokens=1024):
        self._record("chat_stream")
        self.last_messages = messages
        yield from self.stream_parts

    def embed(self, texts, input_type):
        self._record("embed")
        return [fake_vector(t) for t in texts]

    def rerank(self, query, passages):
        self._record("rerank")
        if self.rankings is not None:
            return self.rankings
        return [(i, 5.0 - i) for i in range(len(passages))]
