from rag.chunk import Chunk
from rag.retrieve import retrieve, rewrite_question
from rag.store import VectorStore
from tests.fakes import FakeLLM, fake_vector


def make_chunks(n, doc_id="ac"):
    return [
        Chunk(id=f"{doc_id}:p{10 + i:02d}:0", doc_id=doc_id, doc_name="客廳冷氣", pdf_page=10 + i,
              printed_page=8 + i, section=f"S{i}", text=f"{doc_id} 段落{i}")
        for i in range(n)
    ]


def seeded_store(settings, *groups):
    store = VectorStore(settings.chroma_dir)
    for chunks in groups:
        store.replace_doc(chunks[0].doc_id, chunks, [fake_vector(c.text) for c in chunks])
    return store


def test_rewrite_skipped_without_history():
    llm = FakeLLM(chat_reply="不該用到")
    assert rewrite_question(llm, [], "強力運轉怎麼開？") == "強力運轉怎麼開？"
    assert llm.calls == []


def test_rewrite_uses_recent_history():
    llm = FakeLLM(chat_reply="  強力運轉要怎麼取消？\n")
    history = [(f"舊問題{i}", f"舊回答{i}") for i in range(5)]
    assert rewrite_question(llm, history, "那要怎麼取消？") == "強力運轉要怎麼取消？"
    prompt = llm.last_messages[-1]["content"]
    assert "舊問題4" in prompt and "舊問題2" in prompt
    assert "舊問題1" not in prompt
    assert "那要怎麼取消？" in prompt


def test_rewrite_falls_back_on_empty_reply():
    assert rewrite_question(FakeLLM(chat_reply="  "), [("Q", "A")], "原問題") == "原問題"


def test_retrieve_reranks_and_keeps_top_k(settings):
    store = seeded_store(settings, make_chunks(5))
    llm = FakeLLM(rankings=[(4, 2.0), (0, 1.0), (2, 0.5), (1, -3.0), (3, -4.0)])
    result = retrieve(llm, store, settings, "強力運轉")
    assert result.query == "強力運轉"
    assert len(result.candidates) == 5
    assert [h.chunk for h in result.ranked] == [result.candidates[i].chunk for i in (4, 0, 2)]
    assert [h.vector_rank for h in result.ranked] == [5, 1, 3]
    assert [h.score for h in result.ranked] == [2.0, 1.0, 0.5]
    assert result.found


def test_not_found_when_top_score_below_threshold(settings):
    store = seeded_store(settings, make_chunks(2))
    result = retrieve(FakeLLM(rankings=[(0, -5.0), (1, -6.0)]), store, settings, "烘衣服")
    assert not result.found
    assert len(result.ranked) == 2  # 仍保留給檢索細節顯示


def test_empty_store_returns_not_found_without_rerank(settings):
    llm = FakeLLM()
    result = retrieve(llm, VectorStore(settings.chroma_dir), settings, "任何問題")
    assert not result.found
    assert result.ranked == []
    assert "rerank" not in llm.calls


def test_doc_id_filter(settings):
    store = seeded_store(settings, make_chunks(2, "ac"), make_chunks(2, "bed"))
    result = retrieve(FakeLLM(), store, settings, "問題", doc_id="bed")
    assert {h.chunk.doc_id for h in result.candidates} == {"bed"}
