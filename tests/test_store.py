from rag.chunk import Chunk
from rag.store import VectorStore


def make_chunk(chunk_id, doc_id="ac", printed=9):
    return Chunk(
        id=chunk_id, doc_id=doc_id, doc_name="客廳冷氣", pdf_page=11,
        printed_page=printed, section="強力運轉", text=f"text {chunk_id}",
    )


def test_query_returns_nearest_with_metadata(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.replace_doc("ac", [make_chunk("a"), make_chunk("b", printed=None)], [[1, 0, 0], [0, 1, 0]])
    hits = store.query([0.9, 0.1, 0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "b"]
    assert hits[0].chunk == make_chunk("a")
    assert hits[1].chunk.printed_page is None
    assert hits[0].distance < hits[1].distance


def test_replace_doc_only_touches_that_doc(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.replace_doc("ac", [make_chunk("a"), make_chunk("b")], [[1, 0, 0], [0, 1, 0]])
    store.replace_doc("bed", [make_chunk("c", doc_id="bed")], [[0, 0, 1]])
    store.replace_doc("ac", [make_chunk("a2")], [[1, 0, 0]])
    assert store.count() == 2
    assert {h.chunk.id for h in store.query([1, 1, 1], k=10)} == {"a2", "c"}


def test_query_filters_by_doc_id(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.replace_doc("ac", [make_chunk("a")], [[1, 0, 0]])
    store.replace_doc("bed", [make_chunk("c", doc_id="bed")], [[0, 0, 1]])
    assert [h.chunk.id for h in store.query([1, 0, 0], k=10, doc_id="bed")] == ["c"]


def test_query_on_empty_store_returns_empty(tmp_path):
    assert VectorStore(tmp_path / "chroma").query([1, 0, 0], k=5) == []


def test_data_persists_across_instances(tmp_path):
    VectorStore(tmp_path / "chroma").replace_doc("ac", [make_chunk("a")], [[1, 0, 0]])
    assert VectorStore(tmp_path / "chroma").count() == 1
