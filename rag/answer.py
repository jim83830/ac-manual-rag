"""生成：把檢索到的段落組成 prompt 交給 LLM，並從回答中找出引用的頁面。"""
from __future__ import annotations

import re
from collections.abc import Iterator

from rag.chunk import Chunk
from rag.retrieve import RankedHit

NOT_FOUND_MESSAGE = "說明書裡找不到相關內容。可以換個說法再問一次，或聯絡原廠客服。"

SYSTEM_PROMPT = """你是家電說明書小幫手，回答家人關於家電操作的問題。規則：
1. 只能根據使用者提供的「說明書段落」回答；段落裡沒有的內容，就說「說明書裡沒有提到」，不要自己推測或補充。
2. 一律使用繁體中文，語氣簡單親切。
3. 操作步驟用 1. 2. 3. 條列。
4. 按鍵名稱保留【】格式，例如【快速】。
5. 每個重點後面用括號標註來源頁碼，照抄段落標示的寫法，例如（第 9 頁）或（PDF 第 11 頁）。
6. 如果問題涉及拆機、漏水、電線、異味或冒煙，提醒先停止使用並聯絡專業人員。"""

_PDF_PAGE_RE = re.compile(r"PDF\s*第\s*(\d+)\s*頁")
_PRINTED_PAGE_RE = re.compile(r"第\s*(\d+)\s*頁")


def page_label(chunk: Chunk) -> str:
    if chunk.printed_page is not None:
        return f"第 {chunk.printed_page} 頁"
    return f"PDF 第 {chunk.pdf_page} 頁"


def build_messages(question: str, ranked: list[RankedHit]) -> list[dict]:
    context = "\n\n".join(
        f"[段落 {i}]（{hit.chunk.doc_name}，{page_label(hit.chunk)}）\n{hit.chunk.text}"
        for i, hit in enumerate(ranked, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"說明書段落：\n\n{context}\n\n問題：{question}"},
    ]


def stream_answer(llm, question: str, ranked: list[RankedHit]) -> Iterator[str]:
    yield from llm.chat_stream(build_messages(question, ranked))


def cited_pages(answer: str, ranked: list[RankedHit]) -> list[tuple[str, int]]:
    """回答中引用的頁面 → (doc_id, pdf_page)。只對應本次檢索到的段落；都對不上時用第一名。"""
    if not ranked:
        return []
    mentions = [(m.start(), "pdf", int(m.group(1))) for m in _PDF_PAGE_RE.finditer(answer)]
    # 把「PDF 第 N 頁」挖空，避免再被當成印刷頁碼
    masked = _PDF_PAGE_RE.sub(lambda m: " " * len(m.group(0)), answer)
    mentions += [(m.start(), "printed", int(m.group(1))) for m in _PRINTED_PAGE_RE.finditer(masked)]

    pages: list[tuple[str, int]] = []
    for _, kind, number in sorted(mentions):
        for hit in ranked:
            chunk = hit.chunk
            page = chunk.pdf_page if kind == "pdf" else chunk.printed_page
            key = (chunk.doc_id, chunk.pdf_page)
            if page == number and key not in pages:
                pages.append(key)
    if not pages:
        top = ranked[0].chunk
        pages.append((top.doc_id, top.pdf_page))
    return pages
