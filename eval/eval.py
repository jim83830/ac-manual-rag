"""檢索品質評估：python -m eval.eval [--questions 路徑]

只評檢索與拒答，不評回答文字。
- 檢索命中率：應答題的正解頁，有出現在 rerank 後前幾名，而且有通過門檻
- 拒答正確率：說明書沒有的題目，最高分低於門檻（會回「找不到」）
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.config import ConfigError, load_settings
from rag.llm import NvidiaClient
from rag.retrieve import RetrievalResult, retrieve
from rag.store import VectorStore


@dataclass(frozen=True)
class Question:
    q: str
    expect: list[tuple[str, int]]


@dataclass(frozen=True)
class Outcome:
    question: Question
    found: bool
    top_score: float | None
    pages: list[tuple[str, int]]
    correct: bool


def load_questions(path: Path) -> list[Question]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [
        Question(q=str(e["q"]), expect=[(str(x["doc_id"]), int(x["pdf_page"])) for x in e.get("expect") or []])
        for e in entries
    ]


def judge(question: Question, result: RetrievalResult) -> Outcome:
    pages = [(h.chunk.doc_id, h.chunk.pdf_page) for h in result.ranked]
    if question.expect:
        correct = result.found and any(page in pages for page in question.expect)
    else:
        correct = not result.found
    top_score = result.ranked[0].score if result.ranked else None
    return Outcome(question=question, found=result.found, top_score=top_score, pages=pages, correct=correct)


def _rate(label: str, outcomes: list[Outcome]) -> str:
    if not outcomes:
        return f"{label} —（沒有這類題目）"
    hits = sum(o.correct for o in outcomes)
    return f"{label} {hits}/{len(outcomes)} = {hits * 100 // len(outcomes)}%"


def summarize(outcomes: list[Outcome]) -> str:
    answerable = [o for o in outcomes if o.question.expect]
    unanswerable = [o for o in outcomes if not o.question.expect]
    lines = [_rate("檢索命中率", answerable), _rate("拒答正確率", unanswerable), "", "逐題（分數用來調 RERANK_THRESHOLD）："]
    for o in outcomes:
        kind = "應答" if o.question.expect else "應拒"
        score = f"{o.top_score:6.2f}" if o.top_score is not None else "   —  "
        top = f"p{o.pages[0][1]}" if o.pages else "—"
        expect = ",".join(f"p{p}" for _, p in o.question.expect) or "—"
        lines.append(f"{'✅' if o.correct else '❌'} [{kind}] 分數 {score}  第一名 {top}  正解 {expect}  {o.question.q}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="評估檢索品質")
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.yaml"))
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"設定錯誤：{exc}", file=sys.stderr)
        return 2
    store = VectorStore(settings.chroma_dir)
    if store.count() == 0:
        print("資料庫是空的，請先執行 docker compose run --rm ingest", file=sys.stderr)
        return 1
    llm = NvidiaClient(settings, min_interval=settings.ingest_min_interval)
    outcomes = [judge(q, retrieve(llm, store, settings, q.q)) for q in load_questions(args.questions)]
    print(
        f"設定：chunk={settings.chunk_strategy}，retrieve_k={settings.retrieve_k}，"
        f"rerank_k={settings.rerank_k}，門檻={settings.rerank_threshold}"
    )
    print(summarize(outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
