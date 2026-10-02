import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from rag.llm import LLMError, NvidiaClient


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(kwargs["input"])]
        return SimpleNamespace(data=list(reversed(data)))  # API 不保證回傳順序


def make_client(settings, *, handler=None, openai_client=None, sleeps=None, monotonic=None, min_interval=0.0):
    http = httpx.Client(transport=httpx.MockTransport(handler)) if handler else None
    kwargs = {}
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    return NvidiaClient(
        settings,
        openai_client=openai_client or SimpleNamespace(),
        http_client=http,
        min_interval=min_interval,
        sleep=sleeps.append if sleeps is not None else (lambda s: None),
        **kwargs,
    )


def test_embed_batches_and_keeps_order(settings):
    embeddings = FakeEmbeddings()
    client = make_client(settings, openai_client=SimpleNamespace(embeddings=embeddings))
    texts = ["a" * i for i in range(1, 71)]
    vectors = client.embed(texts, "passage")
    assert vectors == [[float(i)] for i in range(1, 71)]
    assert [len(c["input"]) for c in embeddings.calls] == [32, 32, 6]
    assert embeddings.calls[0]["extra_body"]["input_type"] == "passage"
    assert embeddings.calls[0]["model"] == settings.embed_model


def test_rerank_sorts_by_score(settings):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"rankings": [{"index": 1, "logit": -2.0}, {"index": 0, "logit": 3.5}]})

    client = make_client(settings, handler=handler)
    assert client.rerank("強力運轉", ["甲", "乙"]) == [(0, 3.5), (1, -2.0)]
    assert seen["body"]["query"] == {"text": "強力運轉"}
    assert seen["body"]["passages"] == [{"text": "甲"}, {"text": "乙"}]
    assert seen["body"]["model"] == settings.rerank_model
    assert seen["auth"] == "Bearer test-key"


def test_rerank_retries_on_429(settings):
    statuses = iter([429, 429, 200])

    def handler(request):
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json={"rankings": [{"index": 0, "logit": 1.0}]})
        return httpx.Response(status)

    sleeps = []
    client = make_client(settings, handler=handler, sleeps=sleeps)
    assert client.rerank("q", ["p"]) == [(0, 1.0)]
    assert sleeps == [1, 2]


def test_rerank_fails_fast_on_401(settings):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, text="unauthorized")

    client = make_client(settings, handler=handler)
    with pytest.raises(LLMError, match="401"):
        client.rerank("q", ["p"])
    assert len(calls) == 1


def test_rerank_gives_up_after_retries(settings):
    sleeps = []
    client = make_client(settings, handler=lambda r: httpx.Response(503), sleeps=sleeps)
    with pytest.raises(LLMError, match="重試 5 次"):
        client.rerank("q", ["p"])
    assert sleeps == [1, 2, 4, 8]


def test_rerank_empty_passages_skips_api(settings):
    def handler(request):
        raise AssertionError("不應該呼叫 API")

    assert make_client(settings, handler=handler).rerank("q", []) == []


def test_openai_errors_become_llm_error(settings):
    def boom(**kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    client = make_client(settings, openai_client=fake)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_min_interval_spaces_out_calls(settings):
    times = iter([10.0, 10.5])
    sleeps = []
    client = make_client(
        settings,
        openai_client=SimpleNamespace(embeddings=FakeEmbeddings()),
        sleeps=sleeps,
        monotonic=lambda: next(times),
        min_interval=1.5,
    )
    client.embed(["a"], "query")
    client.embed(["b"], "query")
    assert sleeps == [1.0]
