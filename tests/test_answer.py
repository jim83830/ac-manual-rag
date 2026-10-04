from rag.answer import build_messages, cited_pages, page_label, stream_answer
from rag.chunk import Chunk
from rag.retrieve import RankedHit
from tests.fakes import FakeLLM


def ranked_hit(pdf_page, printed, doc_id="ac"):
    chunk = Chunk(id=f"{doc_id}:{pdf_page}", doc_id=doc_id, doc_name="客廳冷氣", pdf_page=pdf_page,
                  printed_page=printed, section="S", text="客廳冷氣 > S\n內容")
    return RankedHit(chunk=chunk, score=1.0, vector_rank=1)


def test_page_label():
    assert page_label(ranked_hit(11, 9).chunk) == "第 9 頁"
    assert page_label(ranked_hit(3, None).chunk) == "PDF 第 3 頁"


def test_build_messages_includes_labeled_context():
    messages = build_messages("強力運轉怎麼開？", [ranked_hit(11, 9), ranked_hit(3, None)])
    assert messages[0]["role"] == "system"
    assert "只能根據" in messages[0]["content"]
    user = messages[1]["content"]
    assert "[段落 1]（客廳冷氣，第 9 頁）" in user
    assert "[段落 2]（客廳冷氣，PDF 第 3 頁）" in user
    assert user.endswith("問題：強力運轉怎麼開？")


def test_stream_answer_passes_through_parts():
    llm = FakeLLM(stream_parts=["按", "【快速】"])
    assert "".join(stream_answer(llm, "Q", [ranked_hit(11, 9)])) == "按【快速】"
    assert llm.last_messages[0]["role"] == "system"


def test_cited_pages_maps_printed_page_to_pdf_page():
    assert cited_pages("按【快速】鍵（第 9 頁）", [ranked_hit(11, 9), ranked_hit(12, 10)]) == [("ac", 11)]


def test_cited_pages_pdf_label_is_not_read_as_printed_page():
    ranked = [ranked_hit(3, None), ranked_hit(12, 10), ranked_hit(5, 3)]
    assert cited_pages("見（PDF 第 3 頁）和（第 10 頁）", ranked) == [("ac", 3), ("ac", 12)]


def test_cited_pages_falls_back_to_top_hit():
    assert cited_pages("（第 99 頁）", [ranked_hit(11, 9), ranked_hit(12, 10)]) == [("ac", 11)]


def test_cited_pages_dedupes():
    assert cited_pages("（第 9 頁）……（第 9 頁）", [ranked_hit(11, 9)]) == [("ac", 11)]


def test_cited_pages_empty_ranked():
    assert cited_pages("（第 9 頁）", []) == []
