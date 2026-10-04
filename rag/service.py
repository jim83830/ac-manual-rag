"""一次問答的完整流程：改寫 → 檢索 → 生成 → 找頁面圖，並處理錯誤。介面（app.py）只負責顯示。"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag.answer import NOT_FOUND_MESSAGE, cited_pages, page_label, stream_answer
from rag.config import Settings
from rag.llm import LLMError
from rag.pdf import page_image_path
from rag.retrieve import RetrievalResult, retrieve, rewrite_question
from rag.store import VectorStore

SERVICE_ERROR_MESSAGE = "服務暫時無法使用，請稍後再試。"


@dataclass(frozen=True)
class AskUpdate:
    answer: str
    pages: list[Path] = field(default_factory=list)
    debug: str = ""


def _field(message, name):
    return message.get(name) if isinstance(message, dict) else getattr(message, name, None)


def history_pairs(messages: list) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for message in messages:
        role, content = _field(message, "role"), _field(message, "content")
        if not isinstance(content, str):
            continue
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            pairs.append((pending, content))
            pending = None
    return pairs


def format_debug(result: RetrievalResult) -> str:
    lines = [
        f"**檢索用的問題**：{result.query}",
        "",
        f"**Rerank 後前 {len(result.ranked)} 名**（門檻判斷：{'通過' if result.found else '未通過'}）",
        "",
        "| 名次 | 分數 | 向量名次 | 頁面 | 段落 |",
        "|---|---|---|---|---|",
    ]
    for i, hit in enumerate(result.ranked, 1):
        lines.append(f"| {i} | {hit.score:.2f} | {hit.vector_rank} | {page_label(hit.chunk)} | {hit.chunk.section or '—'} |")
    lines += [
        "",
        f"**向量檢索前 {len(result.candidates)} 名**",
        "",
        "| 名次 | 距離 | 頁面 | 段落 |",
        "|---|---|---|---|",
    ]
    for i, hit in enumerate(result.candidates, 1):
        lines.append(f"| {i} | {hit.distance:.3f} | {page_label(hit.chunk)} | {hit.chunk.section or '—'} |")
    return "\n".join(lines)


def ask(llm, store: VectorStore, settings: Settings, question: str, history: list[tuple[str, str]]) -> Iterator[AskUpdate]:
    question = question.strip()
    if not question:
        return
    try:
        query = rewrite_question(llm, history, question)
        result = retrieve(llm, store, settings, query)
    except LLMError:
        yield AskUpdate(answer=SERVICE_ERROR_MESSAGE)
        return

    debug = format_debug(result)
    if not result.found:
        yield AskUpdate(answer=NOT_FOUND_MESSAGE, debug=debug)
        return

    answer = ""
    try:
        for delta in stream_answer(llm, query, result.ranked):
            answer += delta
            yield AskUpdate(answer=answer, debug=debug)
    except LLMError:
        yield AskUpdate(answer=f"{answer}\n\n{SERVICE_ERROR_MESSAGE}" if answer else SERVICE_ERROR_MESSAGE, debug=debug)
        return

    pages = [page_image_path(settings.pages_dir, doc_id, pdf_page) for doc_id, pdf_page in cited_pages(answer, result.ranked)]
    yield AskUpdate(answer=answer, pages=[p for p in pages if p.exists()], debug=debug)
