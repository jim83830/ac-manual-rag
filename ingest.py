"""建索引：manuals/ 裡登記的說明書 → 頁面圖 → OCR → chunk → embedding → Chroma。

用法：docker compose run --rm ingest [--force] [--doc DOC_ID]
"""
from __future__ import annotations

import argparse
import sys

from rag.chunk import Chunk, chunk_page
from rag.config import ConfigError, Settings, load_settings
from rag.llm import LLMError, NvidiaClient
from rag.manuals import Manual, load_manuals
from rag.ocr import OcrReport, ocr_pages, parse_page_doc
from rag.pdf import render_pages
from rag.store import VectorStore


def load_chunks(manual: Manual, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md_path in sorted((settings.ocr_dir / manual.doc_id).glob("p*.md")):
        pdf_page = int(md_path.stem[1:])
        page = parse_page_doc(md_path.read_text(encoding="utf-8"))
        chunks += chunk_page(
            manual.doc_id, manual.name, pdf_page, page, settings.chunk_strategy, settings.chunk_max_chars
        )
    return chunks


def ingest_manual(
    manual: Manual, settings: Settings, llm, store: VectorStore, force: bool = False
) -> tuple[OcrReport, int]:
    pages = render_pages(settings.manuals_dir / manual.file, settings.pages_dir / manual.doc_id, settings.render_dpi)
    report = ocr_pages(pages, settings.ocr_dir / manual.doc_id, llm, settings, force=force)
    chunks = load_chunks(manual, settings)
    embeddings = llm.embed([c.text for c in chunks], "passage") if chunks else []
    store.replace_doc(manual.doc_id, chunks, embeddings)  # embedding 成功才替換，失敗時舊資料不動
    return report, len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 manuals/ 裡登記的說明書建成可搜尋的資料庫")
    parser.add_argument("--force", action="store_true", help="重新 OCR 所有頁面（會覆蓋人工修正）")
    parser.add_argument("--doc", help="只處理指定的 doc_id")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        manuals = load_manuals(settings.manuals_dir)
    except ConfigError as exc:
        print(f"設定錯誤：{exc}", file=sys.stderr)
        return 2
    if args.doc:
        manuals = [m for m in manuals if m.doc_id == args.doc]
        if not manuals:
            print(f"manuals.yaml 裡沒有 doc_id={args.doc}", file=sys.stderr)
            return 2

    llm = NvidiaClient(settings, min_interval=settings.ingest_min_interval)
    store = VectorStore(settings.chroma_dir)
    any_failed = False
    for manual in manuals:
        print(f"▶ {manual.name}（{manual.doc_id}）")
        try:
            report, n_chunks = ingest_manual(manual, settings, llm, store, force=args.force)
        except LLMError as exc:
            print(f"  ❌ embedding 失敗，資料庫未更新：{exc}")
            any_failed = True
            continue
        print(
            f"  OCR 新增 {len(report.done)} 頁、沿用 {len(report.skipped)} 頁、"
            f"失敗 {len(report.failed)} 頁；共 {n_chunks} 個 chunk（策略：{settings.chunk_strategy}）"
        )
        for page, error in report.failed.items():
            print(f"  ❌ PDF 第 {page} 頁：{error}")
        any_failed = any_failed or bool(report.failed)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
