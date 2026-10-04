from PIL import Image

from rag.answer import NOT_FOUND_MESSAGE
from rag.chunk import Chunk
from rag.service import SERVICE_ERROR_MESSAGE, AskUpdate, ask, history_pairs
from rag.store import VectorStore
from tests.fakes import FakeLLM, fake_vector

CHUNK = Chunk(id="ac:p11:0", doc_id="ac", doc_name="客廳冷氣", pdf_page=11, printed_page=9,
              section="強力運轉", text="客廳冷氣 > 強力運轉\n運轉中按【快速】鍵")


def make_store(settings, with_image=True):
    store = VectorStore(settings.chroma_dir)
    store.replace_doc("ac", [CHUNK], [fake_vector(CHUNK.text)])
    if with_image:
        image_dir = settings.pages_dir / "ac"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (10, 10), "white").save(image_dir / "p11.png")
    return store


def test_history_pairs_from_gradio_messages():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    assert history_pairs(messages) == [("Q1", "A1")]


def test_ask_streams_answer_and_attaches_cited_page(settings):
    store = make_store(settings)
    llm = FakeLLM(stream_parts=["按【快速】鍵", "（第 9 頁）"], rankings=[(0, 3.0)])
    updates = list(ask(llm, store, settings, "強力運轉怎麼開？", []))
    assert [u.answer for u in updates] == ["按【快速】鍵", "按【快速】鍵（第 9 頁）", "按【快速】鍵（第 9 頁）"]
    assert updates[0].pages == []
    assert updates[-1].pages == [settings.pages_dir / "ac" / "p11.png"]
    assert "強力運轉怎麼開？" in updates[-1].debug


def test_ask_uses_rewritten_question(settings):
    store = make_store(settings)
    llm = FakeLLM(chat_reply="強力運轉要怎麼取消？", stream_parts=["再按一次【快速】"], rankings=[(0, 3.0)])
    updates = list(ask(llm, store, settings, "那要怎麼取消？", [("強力運轉怎麼開？", "按【快速】")]))
    assert "強力運轉要怎麼取消？" in updates[-1].debug
    assert "強力運轉要怎麼取消？" in llm.last_messages[-1]["content"]


def test_ask_not_found_skips_generation(settings):
    store = make_store(settings)
    llm = FakeLLM(rankings=[(0, -5.0)])
    updates = list(ask(llm, store, settings, "冷氣可以烘衣服嗎？", []))
    assert len(updates) == 1
    assert updates[0].answer == NOT_FOUND_MESSAGE
    assert updates[0].pages == []
    assert "chat_stream" not in llm.calls


def test_ask_retrieval_failure_returns_friendly_message(settings):
    llm = FakeLLM(error_on={"rerank"})
    updates = list(ask(llm, make_store(settings), settings, "強力運轉怎麼開？", []))
    assert updates == [AskUpdate(answer=SERVICE_ERROR_MESSAGE, pages=[], debug="")]


def test_ask_stream_failure_keeps_app_alive(settings):
    llm = FakeLLM(rankings=[(0, 3.0)], error_on={"chat_stream"})
    updates = list(ask(llm, make_store(settings), settings, "強力運轉怎麼開？", []))
    assert updates[-1].answer == SERVICE_ERROR_MESSAGE


def test_ask_ignores_blank_question(settings):
    llm = FakeLLM()
    assert list(ask(llm, make_store(settings), settings, "   ", [])) == []
    assert llm.calls == []


def test_missing_page_image_is_skipped(settings):
    store = make_store(settings, with_image=False)
    llm = FakeLLM(stream_parts=["（第 9 頁）"], rankings=[(0, 3.0)])
    assert list(ask(llm, store, settings, "Q", []))[-1].pages == []
