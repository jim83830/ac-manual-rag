import gradio as gr

from app import build_app
from rag.store import VectorStore
from tests.fakes import FakeLLM


def test_build_app_constructs_blocks(settings):
    demo = build_app(settings, FakeLLM(), VectorStore(settings.chroma_dir))
    assert isinstance(demo, gr.Blocks)
