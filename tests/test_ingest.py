from ingest import ingest_manual, main
from rag.llm import LLMError
from rag.manuals import Manual
from rag.store import VectorStore
from tests.fakes import FakeLLM, fake_vector, make_pdf


def setup_manual(settings, pages):
    settings.manuals_dir.mkdir(parents=True)
    make_pdf(settings.manuals_dir / "m.pdf", pages)
    return Manual(file="m.pdf", doc_id="ac", name="客廳冷氣")


def test_ingest_end_to_end(settings):
    manual = setup_manual(settings, [False, True])
    store = VectorStore(settings.chroma_dir)
    report, n_chunks = ingest_manual(manual, settings, FakeLLM(ocr_results=["頁碼：3\n## 標題\n內容"]), store)

    assert report.done == [2]
    assert n_chunks == 1
    assert store.count() == 1
    assert (settings.pages_dir / "ac" / "p02.png").exists()
    chunk = store.query(fake_vector("x"), k=1)[0].chunk
    assert (chunk.pdf_page, chunk.printed_page, chunk.section) == (2, 3, "標題")


def test_rerun_keeps_manual_edits_without_reocr(settings):
    manual = setup_manual(settings, [True])
    store = VectorStore(settings.chroma_dir)
    ingest_manual(manual, settings, FakeLLM(ocr_results=["頁碼：3\n## 標題\n內容"]), store)
    (settings.ocr_dir / "ac" / "p01.md").write_text("頁碼：3\r\n## 改過的標題\r\n新內容\r\n", encoding="utf-8")

    llm = FakeLLM()
    report, _ = ingest_manual(manual, settings, llm, store)

    assert report.skipped == [1]
    assert "ocr" not in llm.calls
    assert store.count() == 1
    assert store.query(fake_vector("x"), k=1)[0].chunk.section == "改過的標題"


def test_failed_page_is_reported_and_others_still_indexed(settings):
    manual = setup_manual(settings, [True, True])
    store = VectorStore(settings.chroma_dir)
    llm = FakeLLM(ocr_results=[LLMError("boom"), "頁碼：4\n## B\n內容"])
    report, n_chunks = ingest_manual(manual, settings, llm, store)
    assert list(report.failed) == [1]
    assert report.done == [2]
    assert n_chunks == 1


def test_main_rejects_unknown_doc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    (tmp_path / "manuals").mkdir()
    make_pdf(tmp_path / "manuals" / "m.pdf", [True])
    (tmp_path / "manuals" / "manuals.yaml").write_text(
        "- file: m.pdf\n  doc_id: ac\n  name: 冷氣\n", encoding="utf-8"
    )
    assert main(["--doc", "nope"]) == 2
