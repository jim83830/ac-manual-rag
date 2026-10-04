import pytest

from rag.chunk import chunk_page
from rag.ocr import PageDoc

PAGE = PageDoc(
    printed_page=9,
    category="運轉的方式",
    body="## 強力運轉\n運轉中按【快速】鍵\n\n## 鎖定設定\n按【取消】鍵",
)


def contents(chunks):
    return [c.text.split("\n", 1)[1] for c in chunks]


def test_heading_strategy_splits_by_heading():
    chunks = chunk_page("ac", "客廳冷氣", 11, PAGE, "heading", 800)
    assert [c.section for c in chunks] == ["強力運轉", "鎖定設定"]
    assert chunks[0].text == "客廳冷氣 > 運轉的方式 > 強力運轉\n運轉中按【快速】鍵"
    assert [c.id for c in chunks] == ["ac:p11:0", "ac:p11:1"]
    first = chunks[0]
    assert (first.doc_id, first.doc_name, first.pdf_page, first.printed_page) == ("ac", "客廳冷氣", 11, 9)


def test_text_before_first_heading_is_kept():
    page = PageDoc(printed_page=None, category=None, body="前言文字\n## 標題\n內容")
    chunks = chunk_page("ac", "客廳冷氣", 2, page)
    assert [c.section for c in chunks] == ["", "標題"]
    assert chunks[0].text == "客廳冷氣\n前言文字"


def test_long_section_split_by_paragraph():
    body = "## 長段\n" + "\n\n".join(["甲" * 300, "乙" * 300, "丙" * 300])
    chunks = chunk_page("ac", "客廳冷氣", 5, PageDoc(5, None, body), "heading", 700)
    assert contents(chunks) == ["甲" * 300 + "\n\n" + "乙" * 300, "丙" * 300]
    assert all(c.section == "長段" for c in chunks)


def test_oversized_paragraph_is_hard_split():
    chunks = chunk_page("ac", "客廳冷氣", 5, PageDoc(5, None, "## X\n" + "丁" * 1000), "heading", 400)
    assert [len(c) for c in contents(chunks)] == [400, 400, 200]


def test_page_strategy_makes_one_chunk():
    chunks = chunk_page("ac", "客廳冷氣", 11, PAGE, "page", 800)
    assert len(chunks) == 1
    assert chunks[0].section == "強力運轉"
    assert "## 鎖定設定" in chunks[0].text


def test_fixed_strategy_splits_by_size():
    chunks = chunk_page("ac", "客廳冷氣", 5, PageDoc(5, None, "戊" * 1000), "fixed", 400)
    assert [len(c) for c in contents(chunks)] == [400, 400, 200]
    assert all(c.section == "" for c in chunks)


def test_empty_body_gives_no_chunks():
    assert chunk_page("ac", "客廳冷氣", 1, PageDoc(None, None, "  ")) == []


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_page("ac", "客廳冷氣", 1, PAGE, "bogus")
