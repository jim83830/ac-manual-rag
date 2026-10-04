import dataclasses
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


def test_chat_sends_reasoning_effort(settings):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好"))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = make_client(settings, openai_client=fake)
    assert client.chat([{"role": "user", "content": "hi"}]) == "好"
    list(client.chat_stream([{"role": "user", "content": "hi"}]))
    assert [c["extra_body"] for c in calls] == [{"reasoning_effort": "low"}] * 2


def test_reasoning_effort_can_be_disabled(settings):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好"))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = make_client(dataclasses.replace(settings, chat_reasoning_effort=""), openai_client=fake)
    client.chat([{"role": "user", "content": "hi"}])
    assert calls[0]["extra_body"] is None


def test_ocr_uses_longer_timeout(settings):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="頁碼：9"))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = NvidiaClient(settings, openai_client=SimpleNamespace(), vision_client=fake, http_client=httpx.Client())
    assert client.ocr_page(b"jpeg", "prompt") == "頁碼：9"
    assert calls[0]["model"] == settings.vision_model
    assert calls[0]["timeout"] == settings.ocr_timeout


def fake_chat_client(calls, reply):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_ocr_goes_to_vision_client_and_chat_to_main_client(settings):
    main_calls, vision_calls = [], []
    client = NvidiaClient(
        settings,
        openai_client=fake_chat_client(main_calls, "聊天"),
        vision_client=fake_chat_client(vision_calls, "頁碼：9"),
        http_client=httpx.Client(),
    )
    assert client.ocr_page(b"jpeg", "prompt") == "頁碼：9"
    assert client.chat([{"role": "user", "content": "hi"}]) == "聊天"
    assert [c["model"] for c in vision_calls] == [settings.vision_model]
    assert [c["model"] for c in main_calls] == [settings.chat_model]


def test_vision_client_built_from_vision_settings(settings):
    custom = dataclasses.replace(
        settings, vision_base_url="https://vision.example/v1/", vision_api_key="vision-key"
    )
    client = NvidiaClient(custom, openai_client=SimpleNamespace(), http_client=httpx.Client())
    assert str(client._vision.base_url) == "https://vision.example/v1/"
    assert client._vision.api_key == "vision-key"
    assert client._vision.max_retries == custom.ocr_max_retries


def test_list_models_can_query_vision_provider(settings):
    def models(*ids):
        return SimpleNamespace(models=SimpleNamespace(list=lambda: [SimpleNamespace(id=i) for i in ids]))

    client = NvidiaClient(
        settings, openai_client=models("b", "a"), vision_client=models("gemini-x"), http_client=httpx.Client()
    )
    assert client.list_models() == ["a", "b"]
    assert client.list_models(vision=True) == ["gemini-x"]


def test_vision_client_defaults_to_main_provider_without_retries(settings):
    client = NvidiaClient(settings, openai_client=SimpleNamespace(), http_client=httpx.Client())
    assert str(client._vision.base_url).rstrip("/") == settings.base_url.rstrip("/")
    assert client._vision.api_key == settings.api_key
    assert client._vision.max_retries == 0  # OCR 預設不重試：失敗的頁面留給下次 ingest 補做


def test_ocr_rejects_truncated_output(settings):
    def create(**kwargs):
        choice = SimpleNamespace(message=SimpleNamespace(content="## 長時間不使用時\n● 請在晴天時，進行"), finish_reason="length")
        return SimpleNamespace(choices=[choice])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = NvidiaClient(settings, openai_client=SimpleNamespace(), vision_client=fake, http_client=httpx.Client())
    with pytest.raises(LLMError, match="截斷"):
        client.ocr_page(b"jpeg", "prompt")


def test_ocr_allows_long_output(settings):
    calls = []
    fake = fake_chat_client(calls, "頁碼：9")
    client = NvidiaClient(settings, openai_client=SimpleNamespace(), vision_client=fake, http_client=httpx.Client())
    client.ocr_page(b"jpeg", "prompt")
    assert calls[0]["max_tokens"] >= 16384  # 推理型模型的思考也算在上限內


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
