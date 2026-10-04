import dataclasses
import os
import threading
from io import BytesIO

import pytest
from PIL import Image

from rag.llm import LLMError
from rag.ocr import (
    PageDoc,
    clean_model_output,
    encode_for_ocr,
    ocr_pages,
    parse_page_doc,
)
from rag.pdf import PageImage
from tests.fakes import FakeLLM


def test_clean_strips_code_fence():
    assert clean_model_output("```markdown\n頁碼：9\n## 標題\n```") == "頁碼：9\n## 標題\n"


def test_clean_keeps_plain_text():
    assert clean_model_output("\n頁碼：9\n內容  \n\n") == "頁碼：9\n內容\n"


def test_parse_header():
    doc = parse_page_doc("頁碼：9\n分類：運轉的方式\n\n## 強力運轉\n按【快速】鍵")
    assert doc == PageDoc(printed_page=9, category="運轉的方式", body="## 強力運轉\n按【快速】鍵")


def test_parse_tolerates_crlf_halfwidth_colon_and_no_page():
    doc = parse_page_doc("頁碼: 無\r\n## 標題\r\n內容\r\n")
    assert doc == PageDoc(printed_page=None, category=None, body="## 標題\n內容")


def test_parse_without_header():
    assert parse_page_doc("## 標題\n內容") == PageDoc(printed_page=None, category=None, body="## 標題\n內容")


def test_parse_fullwidth_digits():
    assert parse_page_doc("頁碼：１２\n內容").printed_page == 12


def test_encode_for_ocr_downscales_to_jpeg(tmp_path):
    png = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "white").save(png)
    data = encode_for_ocr(png, max_side=1600, quality=85)
    assert data[:2] == b"\xff\xd8"
    assert max(Image.open(BytesIO(data)).size) == 1600


def make_pages(tmp_path, numbers):
    pages = []
    for n in numbers:
        path = tmp_path / f"p{n:02d}.png"
        Image.new("RGB", (50, 50), "white").save(path)
        pages.append(PageImage(pdf_page=n, path=path))
    return pages


def test_ocr_pages_writes_skips_and_reports_failures(tmp_path, settings):
    pages = make_pages(tmp_path, [2, 3, 4])
    out = tmp_path / "ocr"
    out.mkdir()
    (out / "p03.md").write_text("人工修正過的內容\n", encoding="utf-8")
    llm = FakeLLM(ocr_results=["```\n頁碼：1\n內容\n```", LLMError("boom")])

    report = ocr_pages(pages, out, llm, settings)

    assert report.done == [2]
    assert report.skipped == [3]
    assert list(report.failed) == [4]
    assert (out / "p02.md").read_text(encoding="utf-8") == "頁碼：1\n內容\n"
    assert (out / "p03.md").read_text(encoding="utf-8") == "人工修正過的內容\n"
    assert not (out / "p04.md").exists()


class BarrierLLM:
    """兩個 OCR 呼叫必須同時進行，Barrier 才會放行；循序執行會在 timeout 後失敗。"""

    def __init__(self, parties):
        self.barrier = threading.Barrier(parties, timeout=5)

    def ocr_page(self, image_jpeg, prompt):
        self.barrier.wait()
        return "頁碼：1\n內容"


def test_ocr_pages_runs_pages_concurrently(tmp_path, settings):
    pages = make_pages(tmp_path, [2, 3])
    report = ocr_pages(pages, tmp_path / "ocr", BarrierLLM(2), dataclasses.replace(settings, ocr_workers=2))
    assert report.done == [2, 3]
    assert report.failed == {}


def test_interrupted_write_leaves_no_partial_md(tmp_path, settings, monkeypatch):
    def crash(src, dst):
        raise OSError("模擬：寫完暫存檔後、改名前程式中斷")

    monkeypatch.setattr(os, "replace", crash)
    pages = make_pages(tmp_path, [3])
    with pytest.raises(OSError):
        ocr_pages(pages, tmp_path / "ocr", FakeLLM(ocr_results=["頁碼：1\n內容"]), settings)
    assert not (tmp_path / "ocr" / "p03.md").exists()  # 下次重跑才不會誤以為已完成


def test_ocr_pages_force_overwrites(tmp_path, settings):
    pages = make_pages(tmp_path, [3])
    out = tmp_path / "ocr"
    out.mkdir()
    (out / "p03.md").write_text("舊的\n", encoding="utf-8")
    report = ocr_pages(pages, out, FakeLLM(ocr_results=["新的"]), settings, force=True)
    assert report.done == [3]
    assert (out / "p03.md").read_text(encoding="utf-8") == "新的\n"
