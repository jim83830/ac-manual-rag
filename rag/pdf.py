"""PDF → 每頁一張 PNG。圖片同時用於 OCR 與回答時顯示給家人看。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


@dataclass(frozen=True)
class PageImage:
    pdf_page: int  # PDF 頁序，1 起算
    path: Path


def page_filename(pdf_page: int) -> str:
    return f"p{pdf_page:02d}.png"


def page_image_path(pages_dir: Path, doc_id: str, pdf_page: int) -> Path:
    return pages_dir / doc_id / page_filename(pdf_page)


def is_blank(image: Image.Image, white_level: int = 245, min_white_ratio: float = 0.995) -> bool:
    gray = image.convert("L")
    white = sum(gray.histogram()[white_level:])
    return white / (gray.width * gray.height) >= min_white_ratio


def render_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[PageImage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[PageImage] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            image = pdf[index].render(scale=dpi / 72).to_pil()
            if is_blank(image):
                continue
            path = out_dir / page_filename(index + 1)
            image.save(path)
            pages.append(PageImage(pdf_page=index + 1, path=path))
    finally:
        pdf.close()
    return pages
