import pytest

from rag.config import ConfigError
from rag.manuals import Manual, load_manuals

ENTRY = "- file: 客廳冷氣.pdf\n  doc_id: living-room-ac\n  name: 客廳冷氣\n"


def write_registry(directory, text):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manuals.yaml").write_text(text, encoding="utf-8")


def test_loads_registry(tmp_path):
    write_registry(tmp_path, ENTRY)
    (tmp_path / "客廳冷氣.pdf").write_bytes(b"%PDF-1.4")
    assert load_manuals(tmp_path) == [Manual(file="客廳冷氣.pdf", doc_id="living-room-ac", name="客廳冷氣")]


def test_missing_registry(tmp_path):
    with pytest.raises(ConfigError, match="manuals.yaml"):
        load_manuals(tmp_path)


def test_missing_pdf(tmp_path):
    write_registry(tmp_path, ENTRY)
    with pytest.raises(ConfigError, match="找不到說明書檔案"):
        load_manuals(tmp_path)


def test_bad_doc_id(tmp_path):
    write_registry(tmp_path, "- file: a.pdf\n  doc_id: Living Room\n  name: 客廳\n")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ConfigError, match="doc_id"):
        load_manuals(tmp_path)


def test_duplicate_doc_id(tmp_path):
    write_registry(tmp_path, ENTRY + ENTRY)
    (tmp_path / "客廳冷氣.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ConfigError, match="重複"):
        load_manuals(tmp_path)


def test_missing_field(tmp_path):
    write_registry(tmp_path, "- file: a.pdf\n  doc_id: a\n")
    with pytest.raises(ConfigError, match="name"):
        load_manuals(tmp_path)
