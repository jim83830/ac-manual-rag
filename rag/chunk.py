"""把一頁 OCR 結果切成 chunk。策略：heading（依標題，預設）、page（整頁）、fixed（固定字數）。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import CHUNK_STRATEGIES
from rag.ocr import PageDoc

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    pdf_page: int
    printed_page: int | None
    section: str
    text: str  # 第一行是脈絡前綴，例如「客廳冷氣 > 運轉的方式 > 強力運轉」


def _split_sections(body: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title, buffer = "", []
    for line in body.split("\n"):
        match = _HEADING_RE.match(line.strip())
        if match:
            content = "\n".join(buffer).strip()
            if content:
                sections.append((title, content))
            title, buffer = match.group(1), []
        else:
            buffer.append(line)
    content = "\n".join(buffer).strip()
    if content:
        sections.append((title, content))
    return sections


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:  # 單一段落過長就硬切
            if current:
                parts.append(current)
                current = ""
            parts.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if current and len(current) + 2 + len(paragraph) > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def chunk_page(
    doc_id: str,
    doc_name: str,
    pdf_page: int,
    page: PageDoc,
    strategy: str = "heading",
    max_chars: int = 800,
) -> list[Chunk]:
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"未知的 chunk 策略：{strategy}")
    body = page.body.strip()
    if not body:
        return []

    if strategy == "heading":
        pieces = [
            (title, part)
            for title, content in _split_sections(body)
            for part in _split_long(content, max_chars)
        ]
    elif strategy == "page":
        sections = _split_sections(body)
        pieces = [(sections[0][0] if sections else "", body)]
    else:
        pieces = [("", body[i:i + max_chars]) for i in range(0, len(body), max_chars)]

    pieces = [(title, content) for title, content in pieces if content.strip()]
    chunks = []
    for n, (title, content) in enumerate(pieces):
        prefix = " > ".join(part for part in (doc_name, page.category, title) if part)
        chunks.append(Chunk(
            id=f"{doc_id}:p{pdf_page:02d}:{n}",
            doc_id=doc_id,
            doc_name=doc_name,
            pdf_page=pdf_page,
            printed_page=page.printed_page,
            section=title,
            text=f"{prefix}\n{content}",
        ))
    return chunks
