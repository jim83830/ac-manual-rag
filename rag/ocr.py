"""頁面圖片 → Markdown 文字（視覺模型），以及讀回 OCR 檔。

OCR 結果存成 data/ocr/<doc_id>/pNN.md，可以人工修正；檔案已存在就不會重做。
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image

from rag.config import Settings
from rag.llm import LLMError
from rag.pdf import PageImage

OCR_PROMPT = """你是說明書轉寫員。請把這張家電說明書掃描頁完整轉寫成 Markdown，規則：
1. 第一行寫「頁碼：N」，N 是頁面底部印的頁碼數字；找不到就寫「頁碼：無」。
2. 第二行寫「分類：XXX」，XXX 是頁面側邊的分類標籤（通常是直排或反白的字）；沒有就省略這一行。
3. 每個大標題寫成「## 標題」。
4. 遙控器或機器上的按鍵圖示，一律寫成【按鍵名稱】，例如【快速】、【取消】、【運轉/停止】。
5. 表格轉成 Markdown 表格。
6. 示意圖寫成「[圖：簡短描述]」。
7. 只輸出轉寫內容：不要加任何說明，不要用 ``` 包起來，不要翻譯或改寫原文，保持繁體中文。"""

_FENCE_RE = re.compile(r"^```[\w-]*[ \t]*\n(.*?)\n?```$", re.S)
_PAGE_RE = re.compile(r"^頁碼\s*[：:]\s*(\S+)$")
_CATEGORY_RE = re.compile(r"^分類\s*[：:]\s*(.+)$")


@dataclass(frozen=True)
class PageDoc:
    printed_page: int | None
    category: str | None
    body: str


@dataclass
class OcrReport:
    done: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)


def ocr_filename(pdf_page: int) -> str:
    return f"p{pdf_page:02d}.md"


def encode_for_ocr(png_path: Path, max_side: int, quality: int) -> bytes:
    with Image.open(png_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def clean_model_output(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def parse_page_doc(text: str) -> PageDoc:
    lines = text.replace("\r\n", "\n").split("\n")
    printed_page: int | None = None
    category: str | None = None
    body_start = 0
    for index, line in enumerate(lines[:4]):
        stripped = line.strip()
        if not stripped:
            body_start = index + 1
            continue
        if match := _PAGE_RE.match(stripped):
            value = match.group(1)
            printed_page = int(value) if value.isdigit() else None
            body_start = index + 1
            continue
        if match := _CATEGORY_RE.match(stripped):
            category = match.group(1).strip()
            body_start = index + 1
            continue
        break
    body = "\n".join(lines[body_start:]).strip()
    return PageDoc(printed_page=printed_page, category=category, body=body)


def _write_atomic(target: Path, text: str) -> None:
    """先寫暫存檔再改名：中途中斷時只會留下 .tmp，不會出現寫一半的 .md 被誤當成已完成。"""
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def _ocr_one(page: PageImage, llm, settings: Settings) -> tuple[str | None, str | None]:
    """回傳 (文字, 錯誤訊息)，兩者只會有一個。"""
    try:
        image = encode_for_ocr(page.path, settings.ocr_max_side, settings.ocr_jpeg_quality)
        return llm.ocr_page(image, OCR_PROMPT), None
    except (LLMError, OSError) as exc:  # 一頁失敗不影響其他頁
        return None, str(exc)


def ocr_pages(pages: list[PageImage], out_dir: Path, llm, settings: Settings, force: bool = False) -> OcrReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = OcrReport()
    todo: list[PageImage] = []
    for page in pages:
        if (out_dir / ocr_filename(page.pdf_page)).exists() and not force:
            report.skipped.append(page.pdf_page)
        else:
            todo.append(page)

    # 等回應時執行緒會放掉 GIL，所以多執行緒能讓多頁同時等
    with ThreadPoolExecutor(max_workers=max(1, settings.ocr_workers)) as pool:
        results = pool.map(lambda p: _ocr_one(p, llm, settings), todo)
        for page, (text, error) in zip(todo, results):
            if error is not None:
                report.failed[page.pdf_page] = error
                continue
            _write_atomic(out_dir / ocr_filename(page.pdf_page), clean_model_output(text))
            report.done.append(page.pdf_page)
    return report
