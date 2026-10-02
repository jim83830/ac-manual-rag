import pytest

from rag.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key="test-key", manuals_dir=tmp_path / "manuals", data_dir=tmp_path / "data")
