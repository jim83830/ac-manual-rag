import pytest

from rag.config import Settings


@pytest.fixture
def settings(tmp_path):
    # ocr_workers=1：FakeLLM 依呼叫順序回傳結果，並行會讓順序不固定
    return Settings(api_key="test-key", manuals_dir=tmp_path / "manuals", data_dir=tmp_path / "data", ocr_workers=1)
