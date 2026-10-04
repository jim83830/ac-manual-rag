"""檢索：（追問時）改寫問題 → 向量檢索 top k → rerank → 門檻判斷。"""
from __future__ import annotations

from dataclasses import dataclass

from rag.chunk import Chunk
from rag.config import Settings
from rag.store import Hit, VectorStore

MAX_HISTORY_TURNS = 3
_MAX_ANSWER_CHARS = 300

REWRITE_PROMPT = """你會看到使用者和家電說明書助理的對話，以及使用者的最新問題。
請把最新問題改寫成「不看前面對話也能理解」的完整問題，補上被省略的主詞（例如功能名稱）。
只輸出改寫後的問題本身：不要回答它，不要加任何說明。"""


@dataclass(frozen=True)
class RankedHit:
    chunk: Chunk
    score: float
    vector_rank: int  # 在向量檢索中的名次，1 起算


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    candidates: list[Hit]
    ranked: list[RankedHit]
    found: bool


def rewrite_question(llm, history: list[tuple[str, str]], question: str) -> str:
    if not history:
        return question
    conversation = "\n".join(
        f"使用者：{user}\n助理：{answer[:_MAX_ANSWER_CHARS]}" for user, answer in history[-MAX_HISTORY_TURNS:]
    )
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"對話：\n{conversation}\n\n最新問題：{question}"},
    ]
    rewritten = llm.chat(messages, max_tokens=200).strip()
    return rewritten or question


def retrieve(llm, store: VectorStore, settings: Settings, query: str, doc_id: str | None = None) -> RetrievalResult:
    [vector] = llm.embed([query], "query")
    candidates = store.query(vector, settings.retrieve_k, doc_id)
    if not candidates:
        return RetrievalResult(query=query, candidates=[], ranked=[], found=False)
    order = llm.rerank(query, [hit.chunk.text for hit in candidates])
    ranked = [
        RankedHit(chunk=candidates[index].chunk, score=score, vector_rank=index + 1)
        for index, score in order[:settings.rerank_k]
    ]
    found = bool(ranked) and ranked[0].score >= settings.rerank_threshold
    return RetrievalResult(query=query, candidates=candidates, ranked=ranked, found=found)
