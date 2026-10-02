# 冷氣說明書 RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做一個本機 Docker 執行的家電說明書問答網頁：掃描版 PDF 經視覺 OCR、切 chunk、embedding 存入 Chroma，家人用手機提問，系統檢索＋rerank 後串流回答並附上說明書原頁圖片。

**Architecture:** 手寫 RAG 管線，每個步驟一個 `rag/` 模組（pdf → ocr → chunk → store；retrieve → answer → service），`ingest.py` 與 `app.py` 只負責依序呼叫。所有模型呼叫集中在 `rag/llm.py`（NVIDIA NIM，OpenAI 相容），測試一律用不連網的 `FakeLLM`。同一個 Docker image 提供 `app`（Gradio）、`ingest`（一次性）、`test`（pytest，掛載原始碼）三個 compose service。

**Tech Stack:** Python 3.12（Docker `python:3.12-slim`）、openai SDK、httpx、chromadb 1.x、gradio 5.x、pypdfium2、Pillow、PyYAML、pytest、Docker Compose

**Spec:** `docs/superpowers/specs/2026-10-02-ac-manual-rag-design.md`

## Global Constraints

- 所有指令在專案根目錄（`C:\Users\jimchiu\Desktop\code\冷氣說明書RAG`）的 PowerShell 執行，**一律透過 `docker compose`**，不使用主機上的 Python
- 依賴只限 `requirements.txt` 列出的套件；**不用 LangChain / LlamaIndex**
- 介面文字、錯誤訊息、提示詞一律繁體中文
- **`*.pdf` 與 `.env` 永遠不進 git**；`.env` 也不可 COPY 進 image（`.dockerignore` 排除）
- 測試不可連網：模型呼叫用 `tests/fakes.py` 的 `FakeLLM`，HTTP 用 `httpx.MockTransport`
- 頁碼規則：檔名、eval 正解用 `pdf_page`（PDF 頁序，1 起算）；回答引用用 `printed_page`（「第 9 頁」），沒有印刷頁碼時用「PDF 第 N 頁」
- Gradio 監聽 `0.0.0.0:7860`
- 每次 commit 訊息最後一行加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`（以 `git commit -m "標題" -m "Co-Authored-By: ..."` 形式）
- 模型型號不寫死在程式邏輯裡，只放在 `rag/config.py` 預設值，可由 `.env` 覆寫
- 標示 **👤 使用者步驟** 的步驟需要使用者本人操作（取得 API key、審閱 OCR、修改題庫、用手機測試），執行者要停下來請使用者做，不可代做或跳過

**與 spec 的小差異（實作細節，不改設計）：**
- 新增 `rag/manuals.py`（登記表讀取）、`rag/service.py`（把檢索＋生成＋錯誤處理包成可測試的函式，`app.py` 只剩介面）、`scripts/probe_models.py`（確認模型可用）
- 來源頁面圖片以「對話框下方的相簿」顯示最新一則回答的頁面（點圖可放大），而非嵌在對話泡泡裡：Gradio 相簿元件較穩定、手機上可放大

## Review Focus

1. **視覺模型輸出不守規矩**：把整頁包在 ```` ```markdown ```` 裡、或頁碼寫成全形數字／半形冒號 → 仍要正確存檔與解析（Task 5 的 `clean_model_output`、`parse_page_doc` 測試）
2. **使用者在 Windows 上手動修正 OCR 檔**：存成 CRLF 換行、刪掉頁碼行 → 重跑 ingest 不可覆蓋修正、解析不可出錯（Task 5 CRLF 測試、Task 8 重跑測試）
3. **還沒跑 ingest 就開 app / 資料庫是空的** → 不可當掉，問問題回「找不到」，畫面提示先 ingest（Task 7 空資料庫測試、Task 9 空資料庫不呼叫 rerank 測試、Task 12 提示）
4. **問答途中 API 失敗或被限流**（rerank 429、串流中斷）→ 介面顯示「服務暫時無法使用」，App 不中斷、已輸出的部分保留（Task 3 重試測試、Task 11 失敗測試）
5. **空白問題、或 LLM 引用了檢索結果以外的頁碼** → 空白問題直接忽略；不存在的頁碼不顯示錯圖，退回第一名的頁面（Task 11 空白測試、Task 10 退回測試）

---

## File Structure

```
冷氣說明書RAG/
├── manuals/
│   ├── 客廳冷氣.pdf          # 不進 git
│   └── manuals.yaml         # 說明書登記表
├── data/                    # ingest 產出，不進 git（bind mount）
├── rag/
│   ├── __init__.py
│   ├── config.py            # Settings、load_settings、ConfigError、EXAMPLE_QUESTIONS、CHUNK_STRATEGIES
│   ├── manuals.py           # Manual、load_manuals
│   ├── llm.py               # NvidiaClient、LLMError
│   ├── pdf.py               # PageImage、render_pages、is_blank、page_filename、page_image_path
│   ├── ocr.py               # OCR_PROMPT、PageDoc、OcrReport、ocr_pages、parse_page_doc、clean_model_output、encode_for_ocr
│   ├── chunk.py             # Chunk、chunk_page
│   ├── store.py             # VectorStore、Hit
│   ├── retrieve.py          # rewrite_question、retrieve、RankedHit、RetrievalResult
│   ├── answer.py            # build_messages、stream_answer、cited_pages、page_label、NOT_FOUND_MESSAGE
│   └── service.py           # ask、AskUpdate、history_pairs、format_debug、SERVICE_ERROR_MESSAGE
├── scripts/probe_models.py  # 檢查模型是否可用
├── ingest.py                # ingest_manual、load_chunks、main
├── app.py                   # build_app、main
├── eval/
│   ├── __init__.py
│   ├── eval.py              # Question、Outcome、load_questions、judge、summarize、main
│   └── questions.yaml
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # settings fixture
│   ├── fakes.py             # FakeLLM、fake_vector、make_pdf
│   └── test_*.py
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

---

### Task 1: Docker 骨架與設定模組

**Files:**
- Create: `requirements.txt`, `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `pytest.ini`, `.env.example`, `rag/__init__.py`, `rag/config.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 無
- Produces:
  - `rag.config.ConfigError(Exception)`
  - `rag.config.CHUNK_STRATEGIES: tuple[str, ...] = ("heading", "page", "fixed")`
  - `rag.config.EXAMPLE_QUESTIONS: tuple[str, ...]`
  - `rag.config.Settings`（frozen dataclass）欄位：`api_key: str`, `base_url: str`, `vision_model: str`, `chat_model: str`, `embed_model: str`, `rerank_model: str`, `rerank_url: str`, `manuals_dir: Path`, `data_dir: Path`, `render_dpi: int`, `ocr_max_side: int`, `ocr_jpeg_quality: int`, `chunk_strategy: str`, `chunk_max_chars: int`, `retrieve_k: int`, `rerank_k: int`, `rerank_threshold: float`, `ingest_min_interval: float`；property `pages_dir`, `ocr_dir`, `chroma_dir: Path`
  - `rag.config.load_settings(env: Mapping[str, str] | None = None) -> Settings`
  - pytest fixture `settings`（`tests/conftest.py`）：`Settings(api_key="test-key", manuals_dir=tmp_path/"manuals", data_dir=tmp_path/"data")`
  - compose service：`app`、`ingest`（entrypoint `python ingest.py`）、`test`（entrypoint `pytest`，掛載整個專案）

- [ ] **Step 1: 建立 requirements、pytest 設定與 Docker 檔案**

`requirements.txt`：
```
openai>=1.40
httpx>=0.27
chromadb>=1.0,<2
gradio>=5.30,<6
pypdfium2>=4.30
pillow>=10.1
pyyaml>=6.0
pytest>=8.0
```

`pytest.ini`：
```ini
[pytest]
testpaths = tests
pythonpath = .
```

`Dockerfile`：
```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# 先只複製 requirements，程式碼改動時才不用重裝套件（善用 layer cache）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
```

`.dockerignore`：
```
.git
.env
data/
*.pdf
__pycache__/
.pytest_cache/
docs/
```

`docker-compose.yml`：
```yaml
# 三個 service 共用同一個 image，只差啟動指令
x-app: &app
  build: .
  image: ac-manual-rag:dev
  # required: false → 還沒建 .env 時 compose 照樣能用；程式啟動時才會提示缺 API key
  env_file:
    - path: .env
      required: false
  volumes:
    - ./manuals:/app/manuals
    - ./data:/app/data
    - ./eval:/app/eval

services:
  # 常駐：docker compose up app
  app:
    <<: *app
    ports:
      - "7860:7860"

  # 一次性：docker compose run --rm ingest [--force] [--doc DOC_ID]
  ingest:
    <<: *app
    entrypoint: ["python", "ingest.py"]
    profiles: ["tools"]

  # 測試：docker compose run --rm test [pytest 參數]
  # 掛載整個專案，改程式碼不用重 build
  test:
    build: .
    image: ac-manual-rag:dev
    volumes:
      - .:/app
    entrypoint: ["pytest"]
    profiles: ["tools"]
```

`.env.example`：
```
# 到 https://build.nvidia.com 登入後，在任一模型頁按「Get API Key」取得
NVIDIA_API_KEY=nvapi-請換成你的key

# 以下可選：換模型或調參數時取消註解
# VISION_MODEL=meta/llama-3.2-90b-vision-instruct
# CHAT_MODEL=meta/llama-3.3-70b-instruct
# EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
# RERANK_MODEL=nvidia/llama-3.2-nv-rerankqa-1b-v2
# RERANK_URL=https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking
# CHUNK_STRATEGY=heading
# CHUNK_MAX_CHARS=800
# RETRIEVE_K=10
# RERANK_K=3
# RERANK_THRESHOLD=-1.0
```

在 `.gitignore` 的 `# Python` 區塊加一行 `.pytest_cache/`。

建立空檔 `rag/__init__.py`、`tests/__init__.py`。

- [ ] **Step 2: Build image**

Run: `docker compose build test`
Expected: 最後出現 `ac-manual-rag:dev  Built`（第一次會下載套件，需要幾分鐘）

- [ ] **Step 3: 寫失敗測試 `tests/test_config.py` 與 `tests/conftest.py`**

`tests/test_config.py`：
```python
from pathlib import Path

import pytest

from rag.config import ConfigError, load_settings


def test_missing_api_key_raises_helpful_error():
    with pytest.raises(ConfigError, match="NVIDIA_API_KEY"):
        load_settings({})


def test_blank_api_key_counts_as_missing():
    with pytest.raises(ConfigError):
        load_settings({"NVIDIA_API_KEY": "   "})


def test_defaults_and_derived_paths():
    s = load_settings({"NVIDIA_API_KEY": "k"})
    assert s.api_key == "k"
    assert s.chunk_strategy == "heading"
    assert s.pages_dir == Path("data/pages")
    assert s.ocr_dir == Path("data/ocr")
    assert s.chroma_dir == Path("data/chroma")


def test_env_overrides():
    s = load_settings({
        "NVIDIA_API_KEY": "k",
        "CHAT_MODEL": "x/y",
        "CHUNK_STRATEGY": "page",
        "RERANK_THRESHOLD": "0.5",
        "RETRIEVE_K": "5",
    })
    assert (s.chat_model, s.chunk_strategy, s.rerank_threshold, s.retrieve_k) == ("x/y", "page", 0.5, 5)


def test_invalid_values_raise():
    with pytest.raises(ConfigError, match="CHUNK_STRATEGY"):
        load_settings({"NVIDIA_API_KEY": "k", "CHUNK_STRATEGY": "bogus"})
    with pytest.raises(ConfigError, match="RETRIEVE_K"):
        load_settings({"NVIDIA_API_KEY": "k", "RETRIEVE_K": "many"})
```

`tests/conftest.py`：
```python
import pytest

from rag.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key="test-key", manuals_dir=tmp_path / "manuals", data_dir=tmp_path / "data")
```

- [ ] **Step 4: 確認測試失敗**

Run: `docker compose run --rm test tests/test_config.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'rag.config'`

- [ ] **Step 5: 實作 `rag/config.py`**

```python
"""集中所有可調整的設定。模型名稱與調參數值都可用環境變數（.env）覆寫，不必改程式。"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CHUNK_STRATEGIES = ("heading", "page", "fixed")

# 介面上的「常見問題」按鈕
EXAMPLE_QUESTIONS = (
    "強力運轉怎麼開？",
    "濾網要怎麼清潔？",
    "怎麼設定定時關機？",
    "遙控器按了都沒反應",
)


class ConfigError(Exception):
    """設定有誤（缺 API key、數值格式錯誤、登記表錯誤等）。"""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    vision_model: str = "meta/llama-3.2-90b-vision-instruct"
    chat_model: str = "meta/llama-3.3-70b-instruct"
    embed_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    rerank_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    rerank_url: str = (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
    )
    manuals_dir: Path = Path("manuals")
    data_dir: Path = Path("data")
    render_dpi: int = 150
    ocr_max_side: int = 1600
    ocr_jpeg_quality: int = 85
    chunk_strategy: str = "heading"
    chunk_max_chars: int = 800
    retrieve_k: int = 10
    rerank_k: int = 3
    rerank_threshold: float = -1.0
    ingest_min_interval: float = 1.5  # 免費額度約 40 次/分鐘

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def ocr_dir(self) -> Path:
        return self.data_dir / "ocr"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


_STR_ENV = {
    "vision_model": "VISION_MODEL",
    "chat_model": "CHAT_MODEL",
    "embed_model": "EMBED_MODEL",
    "rerank_model": "RERANK_MODEL",
    "rerank_url": "RERANK_URL",
    "chunk_strategy": "CHUNK_STRATEGY",
}
_INT_ENV = {"chunk_max_chars": "CHUNK_MAX_CHARS", "retrieve_k": "RETRIEVE_K", "rerank_k": "RERANK_K"}
_FLOAT_ENV = {"rerank_threshold": "RERANK_THRESHOLD"}


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    api_key = env.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "找不到 NVIDIA_API_KEY：請把 .env.example 複製成 .env，填入 build.nvidia.com 取得的 API key"
        )
    values: dict[str, object] = {"api_key": api_key}
    for field_name, var in _STR_ENV.items():
        if env.get(var, "").strip():
            values[field_name] = env[var].strip()
    for field_name, var in _INT_ENV.items():
        if env.get(var, "").strip():
            try:
                values[field_name] = int(env[var])
            except ValueError:
                raise ConfigError(f"{var} 必須是整數，目前是 {env[var]!r}") from None
    for field_name, var in _FLOAT_ENV.items():
        if env.get(var, "").strip():
            try:
                values[field_name] = float(env[var])
            except ValueError:
                raise ConfigError(f"{var} 必須是數字，目前是 {env[var]!r}") from None
    settings = Settings(**values)
    if settings.chunk_strategy not in CHUNK_STRATEGIES:
        raise ConfigError(
            f"CHUNK_STRATEGY 必須是 {'/'.join(CHUNK_STRATEGIES)} 其中之一，目前是 {settings.chunk_strategy!r}"
        )
    return settings
```

- [ ] **Step 6: 確認測試通過**

Run: `docker compose run --rm test tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```powershell
git add requirements.txt Dockerfile .dockerignore docker-compose.yml pytest.ini .env.example .gitignore rag tests
git commit -m "feat: add docker skeleton and settings module" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 說明書登記表

**Files:**
- Create: `rag/manuals.py`, `manuals/manuals.yaml`, `tests/test_manuals.py`
- Move: `客廳冷氣.pdf` → `manuals/客廳冷氣.pdf`（不進 git）

**Interfaces:**
- Consumes: `rag.config.ConfigError`
- Produces:
  - `rag.manuals.Manual`（frozen dataclass）：`file: str`, `doc_id: str`, `name: str`
  - `rag.manuals.load_manuals(manuals_dir: Path) -> list[Manual]`：讀 `manuals_dir/"manuals.yaml"`，格式錯誤、doc_id 不合法或重複、PDF 不存在時拋 `ConfigError`

- [ ] **Step 1: 寫失敗測試 `tests/test_manuals.py`**

```python
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
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_manuals.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.manuals'`

- [ ] **Step 3: 實作 `rag/manuals.py`**

```python
"""讀取 manuals/manuals.yaml：每本說明書的檔名、代號（doc_id）與顯示名稱。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.config import ConfigError

_DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FIELDS = ("file", "doc_id", "name")


@dataclass(frozen=True)
class Manual:
    file: str
    doc_id: str
    name: str


def load_manuals(manuals_dir: Path) -> list[Manual]:
    registry = manuals_dir / "manuals.yaml"
    if not registry.exists():
        raise ConfigError(f"找不到說明書登記表 {registry}")
    entries = yaml.safe_load(registry.read_text(encoding="utf-8")) or []
    if not isinstance(entries, list):
        raise ConfigError("manuals.yaml 的最外層必須是清單（每本說明書一個以 - 開頭的項目）")

    manuals: list[Manual] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ConfigError(f"manuals.yaml 第 {i} 項格式錯誤")
        missing = [key for key in _FIELDS if not str(entry.get(key) or "").strip()]
        if missing:
            raise ConfigError(f"manuals.yaml 第 {i} 項缺少欄位：{', '.join(missing)}")
        manual = Manual(**{key: str(entry[key]).strip() for key in _FIELDS})
        if not _DOC_ID_RE.match(manual.doc_id):
            raise ConfigError(f"doc_id {manual.doc_id!r} 只能用小寫英文、數字和 -")
        if manual.doc_id in seen:
            raise ConfigError(f"doc_id {manual.doc_id!r} 重複")
        if not (manuals_dir / manual.file).exists():
            raise ConfigError(f"找不到說明書檔案 {manuals_dir / manual.file}")
        seen.add(manual.doc_id)
        manuals.append(manual)
    return manuals
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_manuals.py -v`
Expected: 6 passed

- [ ] **Step 5: 搬移 PDF 並建立真正的登記表**

```powershell
New-Item -ItemType Directory -Force manuals
Move-Item 客廳冷氣.pdf manuals/
```

`manuals/manuals.yaml`：
```yaml
# 每本說明書一項。doc_id 只能用小寫英文、數字和 -
- file: 客廳冷氣.pdf
  doc_id: living-room-ac
  name: 客廳冷氣
```

Run: `git status --short --ignored manuals`
Expected: `manuals/manuals.yaml` 為 `??`（未追蹤），`manuals/客廳冷氣.pdf` 為 `!!`（已忽略）

- [ ] **Step 6: Commit**

```powershell
git add rag/manuals.py tests/test_manuals.py manuals/manuals.yaml
git commit -m "feat: add manual registry" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: NVIDIA API 包裝與模型探測

**Files:**
- Create: `rag/llm.py`, `tests/fakes.py`, `tests/test_llm.py`, `scripts/probe_models.py`
- 👤 Create（使用者）: `.env`

**Interfaces:**
- Consumes: `rag.config.Settings`
- Produces:
  - `rag.llm.LLMError(Exception)`
  - `rag.llm.EMBED_BATCH = 32`
  - `rag.llm.NvidiaClient(settings, *, openai_client=None, http_client=None, min_interval: float = 0.0, sleep=time.sleep, monotonic=time.monotonic)`，方法：
    - `ocr_page(image_jpeg: bytes, prompt: str) -> str`
    - `chat(messages: list[dict], max_tokens: int = 1024) -> str`
    - `chat_stream(messages: list[dict], max_tokens: int = 1024) -> Iterator[str]`（逐段 yield 新增的文字）
    - `embed(texts: list[str], input_type: Literal["passage", "query"]) -> list[list[float]]`
    - `rerank(query: str, passages: list[str]) -> list[tuple[int, float]]`（(passage 索引, 分數)，分數由高到低）
    - `list_models() -> list[str]`
    - 所有方法失敗時拋 `LLMError`
  - `tests.fakes.FakeLLM(*, ocr_results=(), chat_reply="", stream_parts=(), rankings=None, error_on=())`：同上五個方法；`calls: list[str]` 記錄呼叫的方法名；`last_messages` 記錄最後一次 chat/chat_stream 的 messages；`error_on` 內的方法名會拋 `LLMError`
  - `tests.fakes.fake_vector(text: str, dim: int = 8) -> list[float]`

- [ ] **Step 1: 寫 `tests/fakes.py`**

```python
"""測試用替身：不連網、結果可預測。"""
from __future__ import annotations

import hashlib

from rag.llm import LLMError


def fake_vector(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255 + 0.01 for b in digest[:dim]]


class FakeLLM:
    def __init__(self, *, ocr_results=(), chat_reply="", stream_parts=(), rankings=None, error_on=()):
        self.ocr_results = list(ocr_results)
        self.chat_reply = chat_reply
        self.stream_parts = list(stream_parts)
        self.rankings = rankings
        self.error_on = set(error_on)
        self.calls: list[str] = []
        self.last_messages: list[dict] | None = None

    def _record(self, name):
        self.calls.append(name)
        if name in self.error_on:
            raise LLMError(f"fake {name} failure")

    def ocr_page(self, image_jpeg, prompt):
        self._record("ocr")
        result = self.ocr_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def chat(self, messages, max_tokens=1024):
        self._record("chat")
        self.last_messages = messages
        return self.chat_reply

    def chat_stream(self, messages, max_tokens=1024):
        self._record("chat_stream")
        self.last_messages = messages
        yield from self.stream_parts

    def embed(self, texts, input_type):
        self._record("embed")
        return [fake_vector(t) for t in texts]

    def rerank(self, query, passages):
        self._record("rerank")
        if self.rankings is not None:
            return self.rankings
        return [(i, 5.0 - i) for i in range(len(passages))]
```

- [ ] **Step 2: 寫失敗測試 `tests/test_llm.py`**

```python
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
```

- [ ] **Step 3: 確認測試失敗**

Run: `docker compose run --rm test tests/test_llm.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.llm'`

- [ ] **Step 4: 實作 `rag/llm.py`**

```python
"""所有模型 API 呼叫都集中在這裡。要換供應商（Gemini、Claude…）只需要改這個檔案。"""
from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from typing import Literal

import httpx
from openai import OpenAI, OpenAIError

from rag.config import Settings

EMBED_BATCH = 32
_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """呼叫模型 API 失敗（已重試仍失敗，或是不可重試的錯誤）。"""


class NvidiaClient:
    def __init__(
        self,
        settings: Settings,
        *,
        openai_client=None,
        http_client: httpx.Client | None = None,
        min_interval: float = 0.0,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        self._s = settings
        # openai SDK 內建 429 / 5xx 的指數退避重試
        self._openai = openai_client or OpenAI(
            base_url=settings.base_url, api_key=settings.api_key, max_retries=5, timeout=120
        )
        self._http = http_client or httpx.Client(timeout=60)
        self._min_interval = min_interval
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_call: float | None = None

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        now = self._monotonic()
        if self._last_call is not None:
            wait = self._last_call + self._min_interval - now
            if wait > 0:
                self._sleep(wait)
                now += wait
        self._last_call = now

    def ocr_page(self, image_jpeg: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_jpeg).decode("ascii")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        return self._complete(self._s.vision_model, messages, max_tokens=4096, temperature=0.0)

    def chat(self, messages: list[dict], max_tokens: int = 1024) -> str:
        return self._complete(self._s.chat_model, messages, max_tokens=max_tokens, temperature=0.2)

    def chat_stream(self, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        self._throttle()
        try:
            stream = self._openai.chat.completions.create(
                model=self._s.chat_model, messages=messages, max_tokens=max_tokens, temperature=0.2, stream=True
            )
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except OpenAIError as exc:
            raise LLMError(f"聊天模型呼叫失敗：{exc}") from exc

    def _complete(self, model: str, messages: list[dict], *, max_tokens: int, temperature: float) -> str:
        self._throttle()
        try:
            response = self._openai.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
        except OpenAIError as exc:
            raise LLMError(f"模型 {model} 呼叫失敗：{exc}") from exc
        return response.choices[0].message.content or ""

    def embed(self, texts: list[str], input_type: Literal["passage", "query"]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            self._throttle()
            try:
                response = self._openai.embeddings.create(
                    model=self._s.embed_model,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "END"},
                )
            except OpenAIError as exc:
                raise LLMError(f"Embedding 呼叫失敗：{exc}") from exc
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        if not passages:
            return []
        payload = {
            "model": self._s.rerank_model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            "truncate": "END",
        }
        data = self._post_json(self._s.rerank_url, payload)
        ranks = [(int(r["index"]), float(r["logit"])) for r in data["rankings"]]
        return sorted(ranks, key=lambda r: r[1], reverse=True)

    def list_models(self) -> list[str]:
        try:
            return sorted(model.id for model in self._openai.models.list())
        except OpenAIError as exc:
            raise LLMError(f"無法列出模型：{exc}") from exc

    def _post_json(self, url: str, payload: dict, attempts: int = 5) -> dict:
        headers = {"Authorization": f"Bearer {self._s.api_key}", "Accept": "application/json"}
        error = ""
        for attempt in range(attempts):
            self._throttle()
            try:
                response = self._http.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                error = f"連線失敗：{exc}"
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in _RETRY_STATUS:
                    raise LLMError(f"HTTP {response.status_code}：{response.text[:200]}")
                error = f"HTTP {response.status_code}"
            if attempt < attempts - 1:
                self._sleep(2 ** attempt)
        raise LLMError(f"重試 {attempts} 次仍失敗（{error}）")
```

- [ ] **Step 5: 確認測試通過**

Run: `docker compose run --rm test tests/test_llm.py -v`
Expected: 8 passed

- [ ] **Step 6: 寫探測腳本 `scripts/probe_models.py`**

```python
"""檢查 .env 設定的模型是否可用。執行：docker compose run --rm app python scripts/probe_models.py"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rag.config import ConfigError, load_settings  # noqa: E402
from rag.llm import LLMError, NvidiaClient  # noqa: E402


def sample_image() -> bytes:
    image = Image.new("RGB", (640, 200), "white")
    ImageDraw.Draw(image).text((30, 60), "PROBE 4729", fill="black", font=ImageFont.load_default(size=64))
    buffer = BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def check(name, fn) -> bool:
    try:
        print(f"✅ {name}：{fn()}")
        return True
    except LLMError as exc:
        print(f"❌ {name}：{exc}")
        return False


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"設定錯誤：{exc}")
        return 1
    llm = NvidiaClient(settings)
    try:
        available = set(llm.list_models())
    except LLMError as exc:
        print(f"❌ 無法列出模型（API key 是否正確？）：{exc}")
        return 1

    for label, model in [("視覺", settings.vision_model), ("聊天", settings.chat_model), ("Embedding", settings.embed_model)]:
        print(f"{'✅' if model in available else '⚠️ 不在模型清單'} {label}模型 {model}")
    vision_candidates = sorted(m for m in available if any(k in m.lower() for k in ("vision", "vl", "vlm")))
    print("清單中的視覺模型候選：", ", ".join(vision_candidates) or "（找不到）")

    results = [
        check("聊天", lambda: llm.chat([{"role": "user", "content": "用一句繁體中文自我介紹"}], max_tokens=80)),
        check("Embedding", lambda: f"維度 {len(llm.embed(['強力運轉怎麼開'], 'query')[0])}"),
        check("Rerank", lambda: llm.rerank("強力運轉怎麼開", ["定時關機的設定方法", "運轉中按快速鍵可進行強力運轉"])),
        check("視覺", lambda: llm.ocr_page(sample_image(), "圖片裡寫了什麼？只輸出文字。")),
    ]
    print("全部通過 🎉" if all(results) else "有項目失敗：依上面訊息在 .env 換模型後再跑一次")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
```

（Rerank 模型走不同主機，不會出現在模型清單中，所以只用實際呼叫來檢查。）

- [ ] **Step 7: 👤 使用者步驟：取得 API key、建立 `.env`**

請使用者：
1. 到 https://build.nvidia.com 登入（免費加入 NVIDIA Developer Program）
2. 打開任一模型頁面，按「Get API Key」，複製 `nvapi-` 開頭的 key
3. 在專案根目錄執行 `Copy-Item .env.example .env`，用編輯器打開 `.env`，把 `NVIDIA_API_KEY=` 後面換成自己的 key

確認 `.env` 沒有被 git 追蹤：`git status --short`，輸出中**不應**出現 `.env`。

- [ ] **Step 8: 執行探測**

Run: `docker compose build app; docker compose run --rm app python scripts/probe_models.py`
Expected: 四項都是 ✅，最後一行「全部通過 🎉」；視覺那一項的回答包含 `PROBE 4729`

若有 ❌：
- 「不在模型清單」或 404 → 從清單中挑一個替代模型（視覺模型挑上面印出的候選），寫進 `.env` 的對應變數（例如 `VISION_MODEL=...`），重跑本步驟
- 視覺項目出現 payload 太大之類的錯誤 → 先確認 `ocr_page` 的格式；真正 OCR 時圖片會在 Task 5 縮到 1600px 以內
- 把最後採用的模型型號告訴使用者，並寫回 `rag/config.py` 的預設值（讓其他人 clone 後直接可用）

- [ ] **Step 9: Commit**

```powershell
git add rag/llm.py tests/fakes.py tests/test_llm.py scripts/probe_models.py rag/config.py
git commit -m "feat: add NVIDIA API client and model probe" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: PDF 轉頁面圖片

**Files:**
- Create: `rag/pdf.py`, `tests/test_pdf.py`
- Modify: `tests/fakes.py`（新增 `make_pdf`）

**Interfaces:**
- Consumes: 無
- Produces:
  - `rag.pdf.PageImage`（frozen dataclass）：`pdf_page: int`（1 起算）, `path: Path`
  - `rag.pdf.page_filename(pdf_page: int) -> str` → `"p09.png"`
  - `rag.pdf.page_image_path(pages_dir: Path, doc_id: str, pdf_page: int) -> Path` → `pages_dir/doc_id/"p09.png"`
  - `rag.pdf.is_blank(image: PIL.Image.Image, white_level: int = 245, min_white_ratio: float = 0.995) -> bool`
  - `rag.pdf.render_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[PageImage]`：跳過空白頁，只回傳有內容的頁面
  - `tests.fakes.make_pdf(path: Path, pages: list[bool]) -> None`：`True` 表示有內容（畫黑色方塊）、`False` 表示空白頁

- [ ] **Step 1: 在 `tests/fakes.py` 新增 `make_pdf`**

在檔案開頭 import 區加入 `from PIL import Image, ImageDraw`，並在檔案最後加入：

```python
def make_pdf(path, pages):
    """產生測試用 PDF：pages 中 True 表示有內容的頁面，False 表示空白頁。"""
    images = []
    for has_content in pages:
        image = Image.new("RGB", (595, 842), "white")
        if has_content:
            ImageDraw.Draw(image).rectangle((100, 100, 400, 300), fill="black")
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:])
```

- [ ] **Step 2: 寫失敗測試 `tests/test_pdf.py`**

```python
from PIL import Image, ImageDraw

from rag.pdf import PageImage, is_blank, page_filename, page_image_path, render_pages
from tests.fakes import make_pdf


def test_is_blank_detects_white_page():
    assert is_blank(Image.new("RGB", (100, 100), "white"))


def test_is_blank_false_for_content():
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 60, 60), fill="black")
    assert not is_blank(image)


def test_is_blank_tolerates_scan_specks():
    image = Image.new("RGB", (1000, 1000), "white")
    for x in range(20):
        image.putpixel((x * 7, 500), (0, 0, 0))
    assert is_blank(image)


def test_render_skips_blank_pages(tmp_path):
    pdf = tmp_path / "m.pdf"
    make_pdf(pdf, [False, True])
    out = tmp_path / "out"
    assert render_pages(pdf, out, dpi=72) == [PageImage(pdf_page=2, path=out / "p02.png")]
    assert (out / "p02.png").exists()
    assert not (out / "p01.png").exists()


def test_page_paths(tmp_path):
    assert page_filename(9) == "p09.png"
    assert page_image_path(tmp_path, "ac", 11) == tmp_path / "ac" / "p11.png"
```

- [ ] **Step 3: 確認測試失敗**

Run: `docker compose run --rm test tests/test_pdf.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.pdf'`

- [ ] **Step 4: 實作 `rag/pdf.py`**

```python
"""PDF → 每頁一張 PNG。圖片同時用於 OCR 與回答時顯示給家人看。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


@dataclass(frozen=True)
class PageImage:
    pdf_page: int  # PDF 頁序，1 起算
    path: Path


def page_filename(pdf_page: int) -> str:
    return f"p{pdf_page:02d}.png"


def page_image_path(pages_dir: Path, doc_id: str, pdf_page: int) -> Path:
    return pages_dir / doc_id / page_filename(pdf_page)


def is_blank(image: Image.Image, white_level: int = 245, min_white_ratio: float = 0.995) -> bool:
    gray = image.convert("L")
    white = sum(gray.histogram()[white_level:])
    return white / (gray.width * gray.height) >= min_white_ratio


def render_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[PageImage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[PageImage] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(pdf)):
            image = pdf[index].render(scale=dpi / 72).to_pil()
            if is_blank(image):
                continue
            path = out_dir / page_filename(index + 1)
            image.save(path)
            pages.append(PageImage(pdf_page=index + 1, path=path))
    finally:
        pdf.close()
    return pages
```

- [ ] **Step 5: 確認測試通過**

Run: `docker compose run --rm test tests/test_pdf.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```powershell
git add rag/pdf.py tests/test_pdf.py tests/fakes.py
git commit -m "feat: render PDF pages to PNG and skip blank pages" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 視覺 OCR 與 OCR 檔解析

**Files:**
- Create: `rag/ocr.py`, `tests/test_ocr.py`

**Interfaces:**
- Consumes: `rag.config.Settings`（`ocr_max_side`, `ocr_jpeg_quality`）、`rag.llm.LLMError`、`rag.pdf.PageImage`、`llm.ocr_page(image_jpeg, prompt) -> str`
- Produces:
  - `rag.ocr.OCR_PROMPT: str`
  - `rag.ocr.PageDoc`（frozen dataclass）：`printed_page: int | None`, `category: str | None`, `body: str`
  - `rag.ocr.OcrReport`（dataclass）：`done: list[int]`, `skipped: list[int]`, `failed: dict[int, str]`（key 為 pdf_page）
  - `rag.ocr.ocr_filename(pdf_page: int) -> str` → `"p09.md"`
  - `rag.ocr.encode_for_ocr(png_path: Path, max_side: int, quality: int) -> bytes`（JPEG）
  - `rag.ocr.clean_model_output(text: str) -> str`（去掉 code fence，結尾一個換行）
  - `rag.ocr.parse_page_doc(text: str) -> PageDoc`
  - `rag.ocr.ocr_pages(pages: list[PageImage], out_dir: Path, llm, settings: Settings, force: bool = False) -> OcrReport`

- [ ] **Step 1: 寫失敗測試 `tests/test_ocr.py`**

```python
from io import BytesIO

from PIL import Image

from rag.llm import LLMError
from rag.ocr import (
    PageDoc,
    clean_model_output,
    encode_for_ocr,
    ocr_pages,
    parse_page_doc,
)
from rag.pdf import PageImage
from tests.fakes import FakeLLM


def test_clean_strips_code_fence():
    assert clean_model_output("```markdown\n頁碼：9\n## 標題\n```") == "頁碼：9\n## 標題\n"


def test_clean_keeps_plain_text():
    assert clean_model_output("\n頁碼：9\n內容  \n\n") == "頁碼：9\n內容\n"


def test_parse_header():
    doc = parse_page_doc("頁碼：9\n分類：運轉的方式\n\n## 強力運轉\n按【快速】鍵")
    assert doc == PageDoc(printed_page=9, category="運轉的方式", body="## 強力運轉\n按【快速】鍵")


def test_parse_tolerates_crlf_halfwidth_colon_and_no_page():
    doc = parse_page_doc("頁碼: 無\r\n## 標題\r\n內容\r\n")
    assert doc == PageDoc(printed_page=None, category=None, body="## 標題\n內容")


def test_parse_without_header():
    assert parse_page_doc("## 標題\n內容") == PageDoc(printed_page=None, category=None, body="## 標題\n內容")


def test_parse_fullwidth_digits():
    assert parse_page_doc("頁碼：１２\n內容").printed_page == 12


def test_encode_for_ocr_downscales_to_jpeg(tmp_path):
    png = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "white").save(png)
    data = encode_for_ocr(png, max_side=1600, quality=85)
    assert data[:2] == b"\xff\xd8"
    assert max(Image.open(BytesIO(data)).size) == 1600


def make_pages(tmp_path, numbers):
    pages = []
    for n in numbers:
        path = tmp_path / f"p{n:02d}.png"
        Image.new("RGB", (50, 50), "white").save(path)
        pages.append(PageImage(pdf_page=n, path=path))
    return pages


def test_ocr_pages_writes_skips_and_reports_failures(tmp_path, settings):
    pages = make_pages(tmp_path, [2, 3, 4])
    out = tmp_path / "ocr"
    out.mkdir()
    (out / "p03.md").write_text("人工修正過的內容\n", encoding="utf-8")
    llm = FakeLLM(ocr_results=["```\n頁碼：1\n內容\n```", LLMError("boom")])

    report = ocr_pages(pages, out, llm, settings)

    assert report.done == [2]
    assert report.skipped == [3]
    assert list(report.failed) == [4]
    assert (out / "p02.md").read_text(encoding="utf-8") == "頁碼：1\n內容\n"
    assert (out / "p03.md").read_text(encoding="utf-8") == "人工修正過的內容\n"
    assert not (out / "p04.md").exists()


def test_ocr_pages_force_overwrites(tmp_path, settings):
    pages = make_pages(tmp_path, [3])
    out = tmp_path / "ocr"
    out.mkdir()
    (out / "p03.md").write_text("舊的\n", encoding="utf-8")
    report = ocr_pages(pages, out, FakeLLM(ocr_results=["新的"]), settings, force=True)
    assert report.done == [3]
    assert (out / "p03.md").read_text(encoding="utf-8") == "新的\n"
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_ocr.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.ocr'`

- [ ] **Step 3: 實作 `rag/ocr.py`**

```python
"""頁面圖片 → Markdown 文字（視覺模型），以及讀回 OCR 檔。

OCR 結果存成 data/ocr/<doc_id>/pNN.md，可以人工修正；檔案已存在就不會重做。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image

from rag.config import Settings
from rag.llm import LLMError
from rag.pdf import PageImage

OCR_PROMPT = """你是說明書轉寫員。請把這張家電說明書掃描頁完整轉寫成 Markdown，規則：
1. 第一行寫「頁碼：N」，N 是頁面底部印的頁碼數字；找不到就寫「頁碼：無」。
2. 第二行寫「分類：XXX」，XXX 是頁面側邊的分類標籤（通常是直排或反白的字）；沒有就省略這一行。
3. 每個大標題寫成「## 標題」。
4. 遙控器或機器上的按鍵圖示，一律寫成【按鍵名稱】，例如【快速】、【取消】、【運轉/停止】。
5. 表格轉成 Markdown 表格。
6. 示意圖寫成「[圖：簡短描述]」。
7. 只輸出轉寫內容：不要加任何說明，不要用 ``` 包起來，不要翻譯或改寫原文，保持繁體中文。"""

_FENCE_RE = re.compile(r"^```[\w-]*[ \t]*\n(.*?)\n?```$", re.S)
_PAGE_RE = re.compile(r"^頁碼\s*[：:]\s*(\S+)$")
_CATEGORY_RE = re.compile(r"^分類\s*[：:]\s*(.+)$")


@dataclass(frozen=True)
class PageDoc:
    printed_page: int | None
    category: str | None
    body: str


@dataclass
class OcrReport:
    done: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)


def ocr_filename(pdf_page: int) -> str:
    return f"p{pdf_page:02d}.md"


def encode_for_ocr(png_path: Path, max_side: int, quality: int) -> bytes:
    with Image.open(png_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def clean_model_output(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def parse_page_doc(text: str) -> PageDoc:
    lines = text.replace("\r\n", "\n").split("\n")
    printed_page: int | None = None
    category: str | None = None
    body_start = 0
    for index, line in enumerate(lines[:4]):
        stripped = line.strip()
        if not stripped:
            body_start = index + 1
            continue
        if match := _PAGE_RE.match(stripped):
            value = match.group(1)
            printed_page = int(value) if value.isdigit() else None
            body_start = index + 1
            continue
        if match := _CATEGORY_RE.match(stripped):
            category = match.group(1).strip()
            body_start = index + 1
            continue
        break
    body = "\n".join(lines[body_start:]).strip()
    return PageDoc(printed_page=printed_page, category=category, body=body)


def ocr_pages(pages: list[PageImage], out_dir: Path, llm, settings: Settings, force: bool = False) -> OcrReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = OcrReport()
    for page in pages:
        target = out_dir / ocr_filename(page.pdf_page)
        if target.exists() and not force:
            report.skipped.append(page.pdf_page)
            continue
        try:
            image = encode_for_ocr(page.path, settings.ocr_max_side, settings.ocr_jpeg_quality)
            text = llm.ocr_page(image, OCR_PROMPT)
        except (LLMError, OSError) as exc:  # 一頁失敗不影響其他頁
            report.failed[page.pdf_page] = str(exc)
            continue
        target.write_text(clean_model_output(text), encoding="utf-8")
        report.done.append(page.pdf_page)
    return report
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_ocr.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```powershell
git add rag/ocr.py tests/test_ocr.py
git commit -m "feat: add vision OCR with editable markdown output" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 切 chunk

**Files:**
- Create: `rag/chunk.py`, `tests/test_chunk.py`

**Interfaces:**
- Consumes: `rag.config.CHUNK_STRATEGIES`、`rag.ocr.PageDoc`
- Produces:
  - `rag.chunk.Chunk`（frozen dataclass）：`id: str`（`"{doc_id}:p{pdf_page:02d}:{n}"`）, `doc_id: str`, `doc_name: str`, `pdf_page: int`, `printed_page: int | None`, `section: str`, `text: str`（含脈絡前綴第一行）
  - `rag.chunk.chunk_page(doc_id: str, doc_name: str, pdf_page: int, page: PageDoc, strategy: str = "heading", max_chars: int = 800) -> list[Chunk]`，未知策略拋 `ValueError`

- [ ] **Step 1: 寫失敗測試 `tests/test_chunk.py`**

```python
import pytest

from rag.chunk import chunk_page
from rag.ocr import PageDoc

PAGE = PageDoc(
    printed_page=9,
    category="運轉的方式",
    body="## 強力運轉\n運轉中按【快速】鍵\n\n## 鎖定設定\n按【取消】鍵",
)


def contents(chunks):
    return [c.text.split("\n", 1)[1] for c in chunks]


def test_heading_strategy_splits_by_heading():
    chunks = chunk_page("ac", "客廳冷氣", 11, PAGE, "heading", 800)
    assert [c.section for c in chunks] == ["強力運轉", "鎖定設定"]
    assert chunks[0].text == "客廳冷氣 > 運轉的方式 > 強力運轉\n運轉中按【快速】鍵"
    assert [c.id for c in chunks] == ["ac:p11:0", "ac:p11:1"]
    first = chunks[0]
    assert (first.doc_id, first.doc_name, first.pdf_page, first.printed_page) == ("ac", "客廳冷氣", 11, 9)


def test_text_before_first_heading_is_kept():
    page = PageDoc(printed_page=None, category=None, body="前言文字\n## 標題\n內容")
    chunks = chunk_page("ac", "客廳冷氣", 2, page)
    assert [c.section for c in chunks] == ["", "標題"]
    assert chunks[0].text == "客廳冷氣\n前言文字"


def test_long_section_split_by_paragraph():
    body = "## 長段\n" + "\n\n".join(["甲" * 300, "乙" * 300, "丙" * 300])
    chunks = chunk_page("ac", "客廳冷氣", 5, PageDoc(5, None, body), "heading", 700)
    assert contents(chunks) == ["甲" * 300 + "\n\n" + "乙" * 300, "丙" * 300]
    assert all(c.section == "長段" for c in chunks)


def test_oversized_paragraph_is_hard_split():
    chunks = chunk_page("ac", "客廳冷氣", 5, PageDoc(5, None, "## X\n" + "丁" * 1000), "heading", 400)
    assert [len(c) for c in contents(chunks)] == [400, 400, 200]


def test_page_strategy_makes_one_chunk():
    chunks = chunk_page("ac", "客廳冷氣", 11, PAGE, "page", 800)
    assert len(chunks) == 1
    assert chunks[0].section == "強力運轉"
    assert "## 鎖定設定" in chunks[0].text


def test_fixed_strategy_splits_by_size():
    chunks = chunk_page("ac", "客廳冷氣", 5, PageDoc(5, None, "戊" * 1000), "fixed", 400)
    assert [len(c) for c in contents(chunks)] == [400, 400, 200]
    assert all(c.section == "" for c in chunks)


def test_empty_body_gives_no_chunks():
    assert chunk_page("ac", "客廳冷氣", 1, PageDoc(None, None, "  ")) == []


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_page("ac", "客廳冷氣", 1, PAGE, "bogus")
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_chunk.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.chunk'`

- [ ] **Step 3: 實作 `rag/chunk.py`**

```python
"""把一頁 OCR 結果切成 chunk。策略：heading（依標題，預設）、page（整頁）、fixed（固定字數）。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import CHUNK_STRATEGIES
from rag.ocr import PageDoc

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    pdf_page: int
    printed_page: int | None
    section: str
    text: str  # 第一行是脈絡前綴，例如「客廳冷氣 > 運轉的方式 > 強力運轉」


def _split_sections(body: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title, buffer = "", []
    for line in body.split("\n"):
        match = _HEADING_RE.match(line.strip())
        if match:
            content = "\n".join(buffer).strip()
            if content:
                sections.append((title, content))
            title, buffer = match.group(1), []
        else:
            buffer.append(line)
    content = "\n".join(buffer).strip()
    if content:
        sections.append((title, content))
    return sections


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:  # 單一段落過長就硬切
            if current:
                parts.append(current)
                current = ""
            parts.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if current and len(current) + 2 + len(paragraph) > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def chunk_page(
    doc_id: str,
    doc_name: str,
    pdf_page: int,
    page: PageDoc,
    strategy: str = "heading",
    max_chars: int = 800,
) -> list[Chunk]:
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"未知的 chunk 策略：{strategy}")
    body = page.body.strip()
    if not body:
        return []

    if strategy == "heading":
        pieces = [
            (title, part)
            for title, content in _split_sections(body)
            for part in _split_long(content, max_chars)
        ]
    elif strategy == "page":
        sections = _split_sections(body)
        pieces = [(sections[0][0] if sections else "", body)]
    else:
        pieces = [("", body[i:i + max_chars]) for i in range(0, len(body), max_chars)]

    pieces = [(title, content) for title, content in pieces if content.strip()]
    chunks = []
    for n, (title, content) in enumerate(pieces):
        prefix = " > ".join(part for part in (doc_name, page.category, title) if part)
        chunks.append(Chunk(
            id=f"{doc_id}:p{pdf_page:02d}:{n}",
            doc_id=doc_id,
            doc_name=doc_name,
            pdf_page=pdf_page,
            printed_page=page.printed_page,
            section=title,
            text=f"{prefix}\n{content}",
        ))
    return chunks
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_chunk.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```powershell
git add rag/chunk.py tests/test_chunk.py
git commit -m "feat: add heading/page/fixed chunking strategies" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 向量資料庫（Chroma）

**Files:**
- Create: `rag/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `rag.chunk.Chunk`
- Produces:
  - `rag.store.Hit`（frozen dataclass）：`chunk: Chunk`, `distance: float`（cosine 距離，越小越像）
  - `rag.store.VectorStore(path: Path)`，方法：
    - `replace_doc(doc_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None`（先刪該 doc_id 全部舊資料再寫入）
    - `query(embedding: list[float], k: int, doc_id: str | None = None) -> list[Hit]`（空資料庫回 `[]`）
    - `count() -> int`

- [ ] **Step 1: 寫失敗測試 `tests/test_store.py`**

```python
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
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_store.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.store'`

- [ ] **Step 3: 實作 `rag/store.py`**

```python
"""Chroma 向量資料庫的讀寫。embedding 由我們自己算好傳進來，不用 Chroma 內建模型。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from rag.chunk import Chunk

COLLECTION = "manual_chunks"
_NO_PAGE = -1  # Chroma metadata 不能存 None


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    distance: float


def _to_metadata(chunk: Chunk) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "doc_name": chunk.doc_name,
        "pdf_page": chunk.pdf_page,
        "printed_page": chunk.printed_page if chunk.printed_page is not None else _NO_PAGE,
        "section": chunk.section,
    }


def _from_record(chunk_id: str, text: str, meta: dict) -> Chunk:
    printed = int(meta["printed_page"])
    return Chunk(
        id=chunk_id,
        doc_id=meta["doc_id"],
        doc_name=meta["doc_name"],
        pdf_page=int(meta["pdf_page"]),
        printed_page=None if printed == _NO_PAGE else printed,
        section=meta["section"],
        text=text,
    )


class VectorStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def replace_doc(self, doc_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 與 embeddings 數量不一致")
        self._collection.delete(where={"doc_id": doc_id})
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_to_metadata(c) for c in chunks],
        )

    def query(self, embedding: list[float], k: int, doc_id: str | None = None) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, total),
            where={"doc_id": doc_id} if doc_id else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(chunk=_from_record(chunk_id, text, meta), distance=float(distance))
            for chunk_id, text, meta, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def count(self) -> int:
        return self._collection.count()
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_store.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```powershell
git add rag/store.py tests/test_store.py
git commit -m "feat: add Chroma vector store" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Ingest 入口，並對真正的說明書跑一次

**Files:**
- Create: `ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Consumes: `load_settings`, `load_manuals`, `Manual`, `NvidiaClient`, `LLMError`, `render_pages`, `ocr_pages`, `OcrReport`, `parse_page_doc`, `chunk_page`, `Chunk`, `VectorStore`
- Produces:
  - `ingest.load_chunks(manual: Manual, settings: Settings) -> list[Chunk]`（讀 `settings.ocr_dir/doc_id/p*.md`，依檔名排序）
  - `ingest.ingest_manual(manual: Manual, settings: Settings, llm, store: VectorStore, force: bool = False) -> tuple[OcrReport, int]`（第二個值是 chunk 數）
  - `ingest.main(argv: list[str] | None = None) -> int`（0 成功、1 有頁面或 embedding 失敗、2 設定錯誤）
  - 指令：`docker compose run --rm ingest [--force] [--doc DOC_ID]`

- [ ] **Step 1: 寫失敗測試 `tests/test_ingest.py`**

```python
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
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_ingest.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'ingest'`

- [ ] **Step 3: 實作 `ingest.py`**

```python
"""建索引：manuals/ 裡登記的說明書 → 頁面圖 → OCR → chunk → embedding → Chroma。

用法：docker compose run --rm ingest [--force] [--doc DOC_ID]
"""
from __future__ import annotations

import argparse
import sys

from rag.chunk import Chunk, chunk_page
from rag.config import ConfigError, Settings, load_settings
from rag.llm import LLMError, NvidiaClient
from rag.manuals import Manual, load_manuals
from rag.ocr import OcrReport, ocr_pages, parse_page_doc
from rag.pdf import render_pages
from rag.store import VectorStore


def load_chunks(manual: Manual, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md_path in sorted((settings.ocr_dir / manual.doc_id).glob("p*.md")):
        pdf_page = int(md_path.stem[1:])
        page = parse_page_doc(md_path.read_text(encoding="utf-8"))
        chunks += chunk_page(
            manual.doc_id, manual.name, pdf_page, page, settings.chunk_strategy, settings.chunk_max_chars
        )
    return chunks


def ingest_manual(
    manual: Manual, settings: Settings, llm, store: VectorStore, force: bool = False
) -> tuple[OcrReport, int]:
    pages = render_pages(settings.manuals_dir / manual.file, settings.pages_dir / manual.doc_id, settings.render_dpi)
    report = ocr_pages(pages, settings.ocr_dir / manual.doc_id, llm, settings, force=force)
    chunks = load_chunks(manual, settings)
    embeddings = llm.embed([c.text for c in chunks], "passage") if chunks else []
    store.replace_doc(manual.doc_id, chunks, embeddings)  # embedding 成功才替換，失敗時舊資料不動
    return report, len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 manuals/ 裡登記的說明書建成可搜尋的資料庫")
    parser.add_argument("--force", action="store_true", help="重新 OCR 所有頁面（會覆蓋人工修正）")
    parser.add_argument("--doc", help="只處理指定的 doc_id")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        manuals = load_manuals(settings.manuals_dir)
    except ConfigError as exc:
        print(f"設定錯誤：{exc}", file=sys.stderr)
        return 2
    if args.doc:
        manuals = [m for m in manuals if m.doc_id == args.doc]
        if not manuals:
            print(f"manuals.yaml 裡沒有 doc_id={args.doc}", file=sys.stderr)
            return 2

    llm = NvidiaClient(settings, min_interval=settings.ingest_min_interval)
    store = VectorStore(settings.chroma_dir)
    any_failed = False
    for manual in manuals:
        print(f"▶ {manual.name}（{manual.doc_id}）")
        try:
            report, n_chunks = ingest_manual(manual, settings, llm, store, force=args.force)
        except LLMError as exc:
            print(f"  ❌ embedding 失敗，資料庫未更新：{exc}")
            any_failed = True
            continue
        print(
            f"  OCR 新增 {len(report.done)} 頁、沿用 {len(report.skipped)} 頁、"
            f"失敗 {len(report.failed)} 頁；共 {n_chunks} 個 chunk（策略：{settings.chunk_strategy}）"
        )
        for page, error in report.failed.items():
            print(f"  ❌ PDF 第 {page} 頁：{error}")
        any_failed = any_failed or bool(report.failed)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 確認測試通過，並跑全部測試**

Run: `docker compose run --rm test tests/test_ingest.py -v`
Expected: 4 passed

Run: `docker compose run --rm test`
Expected: 全部 passed

- [ ] **Step 5: Commit**

```powershell
git add ingest.py tests/test_ingest.py
git commit -m "feat: add ingest pipeline entrypoint" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: 對真正的說明書跑 ingest**

Run: `docker compose build ingest; docker compose run --rm ingest`
Expected: `▶ 客廳冷氣（living-room-ac）`，接著 `OCR 新增 20 頁、沿用 0 頁、失敗 0 頁；共 N 個 chunk`（第 1 頁空白被跳過；N 約 30～60）。OCR 每頁可能要 10～60 秒，全部約數分鐘。

若有失敗頁：直接再跑一次 `docker compose run --rm ingest`（只會補做失敗的頁面）。

- [ ] **Step 7: 檢查 OCR 品質（執行者先自查）**

用 Read 工具並排看 3 頁：`data/pages/living-room-ac/p11.png` 對照 `data/ocr/living-room-ac/p11.md`，再挑一頁有表格、一頁以文字為主的頁面。檢查：
- 第一行 `頁碼：9`（p11）
- 按鍵寫成【快速】、【取消】
- 「強力運轉的內容」表格是 Markdown 表格
- 沒有出現 ``` 或「以下是轉寫結果」之類的多餘文字

若錯誤明顯偏多（例如大量錯字、按鍵沒用【】）：在 `.env` 改 `VISION_MODEL` 為 Task 3 探測到的其他視覺模型候選，執行 `docker compose run --rm ingest --force`，重新比較。

- [ ] **Step 8: 👤 使用者步驟：審閱並人工修正 OCR**

請使用者用 VS Code 開 `data/ocr/living-room-ac/`，對照 `data/pages/living-room-ac/` 的圖片，修正明顯錯誤（特別是按鍵名稱、數字）。改完後執行：

Run: `docker compose run --rm ingest`
Expected: `OCR 新增 0 頁、沿用 20 頁、失敗 0 頁`（修正內容被保留，只重做 chunk 與 embedding）

（`data/` 不進 git，此步驟沒有 commit。）

---

### Task 9: 檢索（問題改寫、向量檢索、rerank、門檻）

**Files:**
- Create: `rag/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `Settings`（`retrieve_k`, `rerank_k`, `rerank_threshold`）、`VectorStore.query`、`Hit`、`Chunk`、`llm.chat`、`llm.embed`、`llm.rerank`
- Produces:
  - `rag.retrieve.REWRITE_PROMPT: str`、`MAX_HISTORY_TURNS = 3`
  - `rag.retrieve.RankedHit`（frozen dataclass）：`chunk: Chunk`, `score: float`, `vector_rank: int`（1 起算）
  - `rag.retrieve.RetrievalResult`（frozen dataclass）：`query: str`, `candidates: list[Hit]`, `ranked: list[RankedHit]`, `found: bool`
  - `rag.retrieve.rewrite_question(llm, history: list[tuple[str, str]], question: str) -> str`
  - `rag.retrieve.retrieve(llm, store: VectorStore, settings: Settings, query: str, doc_id: str | None = None) -> RetrievalResult`

- [ ] **Step 1: 寫失敗測試 `tests/test_retrieve.py`**

```python
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
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_retrieve.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.retrieve'`

- [ ] **Step 3: 實作 `rag/retrieve.py`**

```python
"""檢索：（追問時）改寫問題 → 向量檢索 top k → rerank → 門檻判斷。"""
from __future__ import annotations

from dataclasses import dataclass

from rag.chunk import Chunk
from rag.config import Settings
from rag.store import Hit, VectorStore

MAX_HISTORY_TURNS = 3
_MAX_ANSWER_CHARS = 300

REWRITE_PROMPT = """你會看到使用者和家電說明書助理的對話，以及使用者的最新問題。
請把最新問題改寫成「不看前面對話也能理解」的完整問題，補上被省略的主詞（例如功能名稱）。
只輸出改寫後的問題本身：不要回答它，不要加任何說明。"""


@dataclass(frozen=True)
class RankedHit:
    chunk: Chunk
    score: float
    vector_rank: int  # 在向量檢索中的名次，1 起算


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    candidates: list[Hit]
    ranked: list[RankedHit]
    found: bool


def rewrite_question(llm, history: list[tuple[str, str]], question: str) -> str:
    if not history:
        return question
    conversation = "\n".join(
        f"使用者：{user}\n助理：{answer[:_MAX_ANSWER_CHARS]}" for user, answer in history[-MAX_HISTORY_TURNS:]
    )
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"對話：\n{conversation}\n\n最新問題：{question}"},
    ]
    rewritten = llm.chat(messages, max_tokens=200).strip()
    return rewritten or question


def retrieve(llm, store: VectorStore, settings: Settings, query: str, doc_id: str | None = None) -> RetrievalResult:
    [vector] = llm.embed([query], "query")
    candidates = store.query(vector, settings.retrieve_k, doc_id)
    if not candidates:
        return RetrievalResult(query=query, candidates=[], ranked=[], found=False)
    order = llm.rerank(query, [hit.chunk.text for hit in candidates])
    ranked = [
        RankedHit(chunk=candidates[index].chunk, score=score, vector_rank=index + 1)
        for index, score in order[:settings.rerank_k]
    ]
    found = bool(ranked) and ranked[0].score >= settings.rerank_threshold
    return RetrievalResult(query=query, candidates=candidates, ranked=ranked, found=found)
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_retrieve.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```powershell
git add rag/retrieve.py tests/test_retrieve.py
git commit -m "feat: add retrieval with query rewrite, rerank and threshold" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: 生成回答與引用頁碼

**Files:**
- Create: `rag/answer.py`, `tests/test_answer.py`

**Interfaces:**
- Consumes: `Chunk`、`RankedHit`、`llm.chat_stream`
- Produces:
  - `rag.answer.SYSTEM_PROMPT: str`、`rag.answer.NOT_FOUND_MESSAGE: str`
  - `rag.answer.page_label(chunk: Chunk) -> str` → `"第 9 頁"` 或 `"PDF 第 11 頁"`
  - `rag.answer.build_messages(question: str, ranked: list[RankedHit]) -> list[dict]`
  - `rag.answer.stream_answer(llm, question: str, ranked: list[RankedHit]) -> Iterator[str]`（逐段 yield 新增文字）
  - `rag.answer.cited_pages(answer: str, ranked: list[RankedHit]) -> list[tuple[str, int]]`（`(doc_id, pdf_page)`，依出現順序、不重複；都對不上時回第一名的頁面；`ranked` 空時回 `[]`）

- [ ] **Step 1: 寫失敗測試 `tests/test_answer.py`**

```python
from rag.answer import build_messages, cited_pages, page_label, stream_answer
from rag.chunk import Chunk
from rag.retrieve import RankedHit
from tests.fakes import FakeLLM


def ranked_hit(pdf_page, printed, doc_id="ac"):
    chunk = Chunk(id=f"{doc_id}:{pdf_page}", doc_id=doc_id, doc_name="客廳冷氣", pdf_page=pdf_page,
                  printed_page=printed, section="S", text="客廳冷氣 > S\n內容")
    return RankedHit(chunk=chunk, score=1.0, vector_rank=1)


def test_page_label():
    assert page_label(ranked_hit(11, 9).chunk) == "第 9 頁"
    assert page_label(ranked_hit(3, None).chunk) == "PDF 第 3 頁"


def test_build_messages_includes_labeled_context():
    messages = build_messages("強力運轉怎麼開？", [ranked_hit(11, 9), ranked_hit(3, None)])
    assert messages[0]["role"] == "system"
    assert "只能根據" in messages[0]["content"]
    user = messages[1]["content"]
    assert "[段落 1]（客廳冷氣，第 9 頁）" in user
    assert "[段落 2]（客廳冷氣，PDF 第 3 頁）" in user
    assert user.endswith("問題：強力運轉怎麼開？")


def test_stream_answer_passes_through_parts():
    llm = FakeLLM(stream_parts=["按", "【快速】"])
    assert "".join(stream_answer(llm, "Q", [ranked_hit(11, 9)])) == "按【快速】"
    assert llm.last_messages[0]["role"] == "system"


def test_cited_pages_maps_printed_page_to_pdf_page():
    assert cited_pages("按【快速】鍵（第 9 頁）", [ranked_hit(11, 9), ranked_hit(12, 10)]) == [("ac", 11)]


def test_cited_pages_pdf_label_is_not_read_as_printed_page():
    ranked = [ranked_hit(3, None), ranked_hit(12, 10), ranked_hit(5, 3)]
    assert cited_pages("見（PDF 第 3 頁）和（第 10 頁）", ranked) == [("ac", 3), ("ac", 12)]


def test_cited_pages_falls_back_to_top_hit():
    assert cited_pages("（第 99 頁）", [ranked_hit(11, 9), ranked_hit(12, 10)]) == [("ac", 11)]


def test_cited_pages_dedupes():
    assert cited_pages("（第 9 頁）……（第 9 頁）", [ranked_hit(11, 9)]) == [("ac", 11)]


def test_cited_pages_empty_ranked():
    assert cited_pages("（第 9 頁）", []) == []
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_answer.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.answer'`

- [ ] **Step 3: 實作 `rag/answer.py`**

```python
"""生成：把檢索到的段落組成 prompt 交給 LLM，並從回答中找出引用的頁面。"""
from __future__ import annotations

import re
from collections.abc import Iterator

from rag.chunk import Chunk
from rag.retrieve import RankedHit

NOT_FOUND_MESSAGE = "說明書裡找不到相關內容。可以換個說法再問一次，或聯絡原廠客服。"

SYSTEM_PROMPT = """你是家電說明書小幫手，回答家人關於家電操作的問題。規則：
1. 只能根據使用者提供的「說明書段落」回答；段落裡沒有的內容，就說「說明書裡沒有提到」，不要自己推測或補充。
2. 一律使用繁體中文，語氣簡單親切。
3. 操作步驟用 1. 2. 3. 條列。
4. 按鍵名稱保留【】格式，例如【快速】。
5. 每個重點後面用括號標註來源頁碼，照抄段落標示的寫法，例如（第 9 頁）或（PDF 第 11 頁）。
6. 如果問題涉及拆機、漏水、電線、異味或冒煙，提醒先停止使用並聯絡專業人員。"""

_PDF_PAGE_RE = re.compile(r"PDF\s*第\s*(\d+)\s*頁")
_PRINTED_PAGE_RE = re.compile(r"第\s*(\d+)\s*頁")


def page_label(chunk: Chunk) -> str:
    if chunk.printed_page is not None:
        return f"第 {chunk.printed_page} 頁"
    return f"PDF 第 {chunk.pdf_page} 頁"


def build_messages(question: str, ranked: list[RankedHit]) -> list[dict]:
    context = "\n\n".join(
        f"[段落 {i}]（{hit.chunk.doc_name}，{page_label(hit.chunk)}）\n{hit.chunk.text}"
        for i, hit in enumerate(ranked, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"說明書段落：\n\n{context}\n\n問題：{question}"},
    ]


def stream_answer(llm, question: str, ranked: list[RankedHit]) -> Iterator[str]:
    yield from llm.chat_stream(build_messages(question, ranked))


def cited_pages(answer: str, ranked: list[RankedHit]) -> list[tuple[str, int]]:
    """回答中引用的頁面 → (doc_id, pdf_page)。只對應本次檢索到的段落；都對不上時用第一名。"""
    if not ranked:
        return []
    mentions = [(m.start(), "pdf", int(m.group(1))) for m in _PDF_PAGE_RE.finditer(answer)]
    # 把「PDF 第 N 頁」挖空，避免再被當成印刷頁碼
    masked = _PDF_PAGE_RE.sub(lambda m: " " * len(m.group(0)), answer)
    mentions += [(m.start(), "printed", int(m.group(1))) for m in _PRINTED_PAGE_RE.finditer(masked)]

    pages: list[tuple[str, int]] = []
    for _, kind, number in sorted(mentions):
        for hit in ranked:
            chunk = hit.chunk
            page = chunk.pdf_page if kind == "pdf" else chunk.printed_page
            key = (chunk.doc_id, chunk.pdf_page)
            if page == number and key not in pages:
                pages.append(key)
    if not pages:
        top = ranked[0].chunk
        pages.append((top.doc_id, top.pdf_page))
    return pages
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_answer.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```powershell
git add rag/answer.py tests/test_answer.py
git commit -m "feat: add answer generation and page citation parsing" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: 問答服務（串起檢索與生成、錯誤處理、檢索細節）

**Files:**
- Create: `rag/service.py`, `tests/test_service.py`

**Interfaces:**
- Consumes: `rewrite_question`、`retrieve`、`RetrievalResult`、`stream_answer`、`cited_pages`、`page_label`、`NOT_FOUND_MESSAGE`、`page_image_path`、`LLMError`、`Settings`、`VectorStore`
- Produces:
  - `rag.service.SERVICE_ERROR_MESSAGE: str`
  - `rag.service.AskUpdate`（frozen dataclass）：`answer: str`, `pages: list[Path]`, `debug: str`
  - `rag.service.history_pairs(messages: list) -> list[tuple[str, str]]`（Gradio messages 格式 → (問, 答) 配對）
  - `rag.service.format_debug(result: RetrievalResult) -> str`（Markdown）
  - `rag.service.ask(llm, store, settings, question: str, history: list[tuple[str, str]]) -> Iterator[AskUpdate]`：空白問題不 yield 任何東西；每次 yield 的 `answer` 是到目前為止的完整回答；只有最後一次 yield 帶 `pages`

- [ ] **Step 1: 寫失敗測試 `tests/test_service.py`**

```python
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
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_service.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'rag.service'`

- [ ] **Step 3: 實作 `rag/service.py`**

```python
"""一次問答的完整流程：改寫 → 檢索 → 生成 → 找頁面圖，並處理錯誤。介面（app.py）只負責顯示。"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag.answer import NOT_FOUND_MESSAGE, cited_pages, page_label, stream_answer
from rag.config import Settings
from rag.llm import LLMError
from rag.pdf import page_image_path
from rag.retrieve import RetrievalResult, retrieve, rewrite_question
from rag.store import VectorStore

SERVICE_ERROR_MESSAGE = "服務暫時無法使用，請稍後再試。"


@dataclass(frozen=True)
class AskUpdate:
    answer: str
    pages: list[Path] = field(default_factory=list)
    debug: str = ""


def _field(message, name):
    return message.get(name) if isinstance(message, dict) else getattr(message, name, None)


def history_pairs(messages: list) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for message in messages:
        role, content = _field(message, "role"), _field(message, "content")
        if not isinstance(content, str):
            continue
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            pairs.append((pending, content))
            pending = None
    return pairs


def format_debug(result: RetrievalResult) -> str:
    lines = [
        f"**檢索用的問題**：{result.query}",
        "",
        f"**Rerank 後前 {len(result.ranked)} 名**（門檻判斷：{'通過' if result.found else '未通過'}）",
        "",
        "| 名次 | 分數 | 向量名次 | 頁面 | 段落 |",
        "|---|---|---|---|---|",
    ]
    for i, hit in enumerate(result.ranked, 1):
        lines.append(f"| {i} | {hit.score:.2f} | {hit.vector_rank} | {page_label(hit.chunk)} | {hit.chunk.section or '—'} |")
    lines += [
        "",
        f"**向量檢索前 {len(result.candidates)} 名**",
        "",
        "| 名次 | 距離 | 頁面 | 段落 |",
        "|---|---|---|---|",
    ]
    for i, hit in enumerate(result.candidates, 1):
        lines.append(f"| {i} | {hit.distance:.3f} | {page_label(hit.chunk)} | {hit.chunk.section or '—'} |")
    return "\n".join(lines)


def ask(llm, store: VectorStore, settings: Settings, question: str, history: list[tuple[str, str]]) -> Iterator[AskUpdate]:
    question = question.strip()
    if not question:
        return
    try:
        query = rewrite_question(llm, history, question)
        result = retrieve(llm, store, settings, query)
    except LLMError:
        yield AskUpdate(answer=SERVICE_ERROR_MESSAGE)
        return

    debug = format_debug(result)
    if not result.found:
        yield AskUpdate(answer=NOT_FOUND_MESSAGE, debug=debug)
        return

    answer = ""
    try:
        for delta in stream_answer(llm, query, result.ranked):
            answer += delta
            yield AskUpdate(answer=answer, debug=debug)
    except LLMError:
        yield AskUpdate(answer=f"{answer}\n\n{SERVICE_ERROR_MESSAGE}" if answer else SERVICE_ERROR_MESSAGE, debug=debug)
        return

    pages = [page_image_path(settings.pages_dir, doc_id, pdf_page) for doc_id, pdf_page in cited_pages(answer, result.ranked)]
    yield AskUpdate(answer=answer, pages=[p for p in pages if p.exists()], debug=debug)
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_service.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```powershell
git add rag/service.py tests/test_service.py
git commit -m "feat: add ask service with error handling and debug info" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Gradio 介面

**Files:**
- Create: `app.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `load_settings`、`ConfigError`、`EXAMPLE_QUESTIONS`、`NvidiaClient`、`VectorStore`、`ask`、`history_pairs`
- Produces:
  - `app.build_app(settings: Settings, llm, store: VectorStore) -> gr.Blocks`
  - `app.main() -> None`（`docker compose up app` 的進入點）

- [ ] **Step 1: 寫失敗測試 `tests/test_app.py`**

```python
import gradio as gr

from app import build_app
from rag.store import VectorStore
from tests.fakes import FakeLLM


def test_build_app_constructs_blocks(settings):
    demo = build_app(settings, FakeLLM(), VectorStore(settings.chroma_dir))
    assert isinstance(demo, gr.Blocks)
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_app.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: 實作 `app.py`**

```python
"""家人用的問答網頁（Gradio）。啟動：docker compose up app，手機連 http://<電腦IP>:7860"""
from __future__ import annotations

import sys

import gradio as gr

from rag.config import EXAMPLE_QUESTIONS, ConfigError, Settings, load_settings
from rag.llm import NvidiaClient
from rag.service import ask, history_pairs
from rag.store import VectorStore


def build_app(settings: Settings, llm, store: VectorStore) -> gr.Blocks:
    with gr.Blocks(title="家電說明書小幫手") as demo:
        gr.Markdown("# 🏠 家電說明書小幫手")
        if store.count() == 0:
            gr.Markdown("⚠️ 資料庫是空的，請先執行 `docker compose run --rm ingest`")
        gr.Markdown("常見問題（點一下就問）：")
        with gr.Row():
            example_buttons = [gr.Button(q, size="sm") for q in EXAMPLE_QUESTIONS]
        chatbot = gr.Chatbot(type="messages", label="對話", height=420)
        gallery = gr.Gallery(label="說明書原頁（點圖可放大）", columns=2, height="auto", object_fit="contain")
        with gr.Accordion("🔍 檢索細節", open=False):
            debug = gr.Markdown()
        with gr.Row():
            box = gr.Textbox(placeholder="輸入問題…", show_label=False, scale=4)
            send = gr.Button("送出", variant="primary", scale=1)
        clear = gr.Button("清除對話")

        def respond(question, messages):
            question = (question or "").strip()
            messages = list(messages or [])
            if not question:
                yield messages, gr.update(), gr.update(), ""
                return
            history = history_pairs(messages)
            base = messages + [{"role": "user", "content": question}]
            for update in ask(llm, store, settings, question, history):
                yield (
                    base + [{"role": "assistant", "content": update.answer}],
                    [str(p) for p in update.pages],
                    update.debug,
                    "",
                )

        outputs = [chatbot, gallery, debug, box]
        send.click(respond, [box, chatbot], outputs)
        box.submit(respond, [box, chatbot], outputs)
        for button in example_buttons:
            button.click(respond, [button, chatbot], outputs)  # 按鈕的值就是它的文字
        clear.click(lambda: ([], [], "", ""), None, outputs)
    return demo


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        sys.exit(f"設定錯誤：{exc}")
    store = VectorStore(settings.chroma_dir)
    demo = build_app(settings, NvidiaClient(settings), store)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(settings.data_dir.resolve())],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_app.py -v`
Expected: 1 passed

- [ ] **Step 5: 在電腦上實際跑起來**

Run: `docker compose up --build app`
Expected: log 出現 `Running on local URL:  http://0.0.0.0:7860`

用瀏覽器開 http://localhost:7860，依序測：
1. 點「強力運轉怎麼開？」→ 串流出現回答，含【快速】與（第 9 頁），下方相簿顯示 PDF 第 11 頁圖片，點圖可放大
2. 接著輸入「那要怎麼取消？」→ 展開「🔍 檢索細節」，檢索用的問題應被改寫成包含「強力運轉」
3. 輸入「冷氣可以烘衣服嗎？」→ 回「說明書裡找不到相關內容…」，不顯示圖片
4. 什麼都不打直接按「送出」→ 沒有任何反應、不報錯
5. 按「清除對話」→ 對話、相簿、檢索細節都清空

若第 3 項沒有回「找不到」或第 1 項回「找不到」：先記下檢索細節裡的分數，門檻在 Task 13 用 eval 調整。

- [ ] **Step 6: 👤 使用者步驟：用手機連線**

1. 電腦 PowerShell 執行 `ipconfig`，找「Wi-Fi」介面卡的 IPv4 位址（例如 `192.168.1.23`）
2. 手機連同一個 Wi-Fi，瀏覽器開 `http://192.168.1.23:7860`
3. 若連不上：Windows 防火牆可能擋住，第一次啟動時若跳出 Docker Desktop 的防火牆詢問，要勾選「私人網路」並允許；或到「Windows 安全性 → 防火牆與網路保護 → 允許應用程式通過防火牆」確認 Docker Desktop 在私人網路是允許的
4. 在手機上重複 Step 5 的第 1 項，確認版面可用、圖片可放大

結束時在 PowerShell 按 `Ctrl+C` 停止，或 `docker compose down`。

- [ ] **Step 7: Commit**

```powershell
git add app.py tests/test_app.py
git commit -m "feat: add Gradio chat interface" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: 測試題庫與 Eval

**Files:**
- Create: `eval/__init__.py`, `eval/eval.py`, `eval/questions.yaml`, `tests/test_eval.py`
- Modify: `rag/config.py`（依 eval 結果更新 `rerank_threshold` 預設值、`EXAMPLE_QUESTIONS`）

**Interfaces:**
- Consumes: `load_settings`、`NvidiaClient`、`VectorStore`、`retrieve`、`RetrievalResult`、`RankedHit`
- Produces:
  - `eval.eval.Question`（frozen dataclass）：`q: str`, `expect: list[tuple[str, int]]`（`(doc_id, pdf_page)`；空清單表示說明書沒有）
  - `eval.eval.Outcome`（frozen dataclass）：`question: Question`, `found: bool`, `top_score: float | None`, `pages: list[tuple[str, int]]`, `correct: bool`
  - `eval.eval.load_questions(path: Path) -> list[Question]`
  - `eval.eval.judge(question: Question, result: RetrievalResult) -> Outcome`
  - `eval.eval.summarize(outcomes: list[Outcome]) -> str`
  - `eval.eval.main(argv: list[str] | None = None) -> int`
  - 指令：`docker compose run --rm app python -m eval.eval`

- [ ] **Step 1: 寫失敗測試 `tests/test_eval.py`**

```python
from eval.eval import Outcome, Question, judge, load_questions, summarize
from rag.chunk import Chunk
from rag.retrieve import RankedHit, RetrievalResult


def result(found, pages_scores):
    ranked = [
        RankedHit(chunk=Chunk(id=f"ac:{p}", doc_id="ac", doc_name="客廳冷氣", pdf_page=p, printed_page=None,
                              section="S", text="t"), score=s, vector_rank=1)
        for p, s in pages_scores
    ]
    return RetrievalResult(query="q", candidates=[], ranked=ranked, found=found)


def test_load_questions(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "- q: 強力運轉怎麼開？\n  expect: [{doc_id: ac, pdf_page: 11}]\n- q: 可以烘衣服嗎？\n  expect: []\n",
        encoding="utf-8",
    )
    assert load_questions(path) == [Question("強力運轉怎麼開？", [("ac", 11)]), Question("可以烘衣服嗎？", [])]


def test_judge_answerable_hit():
    outcome = judge(Question("q", [("ac", 11)]), result(True, [(12, 2.0), (11, 1.0)]))
    assert outcome.correct
    assert outcome.top_score == 2.0
    assert outcome.pages == [("ac", 12), ("ac", 11)]


def test_judge_answerable_miss_when_refused():
    assert not judge(Question("q", [("ac", 11)]), result(False, [(11, -3.0)])).correct


def test_judge_unanswerable():
    question = Question("q", [])
    assert judge(question, result(False, [(11, -3.0)])).correct
    assert not judge(question, result(True, [(11, 2.0)])).correct


def test_judge_empty_ranked_has_no_score():
    assert judge(Question("q", []), result(False, [])).top_score is None


def test_summarize_counts():
    outcomes = [
        Outcome(Question("a", [("ac", 1)]), True, 2.0, [("ac", 1)], True),
        Outcome(Question("b", [("ac", 2)]), True, 0.5, [("ac", 3)], False),
        Outcome(Question("c", []), False, -4.0, [("ac", 3)], True),
    ]
    text = summarize(outcomes)
    assert "檢索命中率 1/2 = 50%" in text
    assert "拒答正確率 1/1 = 100%" in text
    assert "❌" in text and "b" in text
```

- [ ] **Step 2: 確認測試失敗**

Run: `docker compose run --rm test tests/test_eval.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'eval.eval'`

- [ ] **Step 3: 實作 `eval/__init__.py`（空檔）與 `eval/eval.py`**

```python
"""檢索品質評估：python -m eval.eval [--questions 路徑]

只評檢索與拒答，不評回答文字。
- 檢索命中率：應答題的正解頁，有出現在 rerank 後前幾名，而且有通過門檻
- 拒答正確率：說明書沒有的題目，最高分低於門檻（會回「找不到」）
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.config import ConfigError, load_settings
from rag.llm import NvidiaClient
from rag.retrieve import RetrievalResult, retrieve
from rag.store import VectorStore


@dataclass(frozen=True)
class Question:
    q: str
    expect: list[tuple[str, int]]


@dataclass(frozen=True)
class Outcome:
    question: Question
    found: bool
    top_score: float | None
    pages: list[tuple[str, int]]
    correct: bool


def load_questions(path: Path) -> list[Question]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [
        Question(q=str(e["q"]), expect=[(str(x["doc_id"]), int(x["pdf_page"])) for x in e.get("expect") or []])
        for e in entries
    ]


def judge(question: Question, result: RetrievalResult) -> Outcome:
    pages = [(h.chunk.doc_id, h.chunk.pdf_page) for h in result.ranked]
    if question.expect:
        correct = result.found and any(page in pages for page in question.expect)
    else:
        correct = not result.found
    top_score = result.ranked[0].score if result.ranked else None
    return Outcome(question=question, found=result.found, top_score=top_score, pages=pages, correct=correct)


def _rate(label: str, outcomes: list[Outcome]) -> str:
    if not outcomes:
        return f"{label} —（沒有這類題目）"
    hits = sum(o.correct for o in outcomes)
    return f"{label} {hits}/{len(outcomes)} = {hits * 100 // len(outcomes)}%"


def summarize(outcomes: list[Outcome]) -> str:
    answerable = [o for o in outcomes if o.question.expect]
    unanswerable = [o for o in outcomes if not o.question.expect]
    lines = [_rate("檢索命中率", answerable), _rate("拒答正確率", unanswerable), "", "逐題（分數用來調 RERANK_THRESHOLD）："]
    for o in outcomes:
        kind = "應答" if o.question.expect else "應拒"
        score = f"{o.top_score:6.2f}" if o.top_score is not None else "   —  "
        top = f"p{o.pages[0][1]}" if o.pages else "—"
        expect = ",".join(f"p{p}" for _, p in o.question.expect) or "—"
        lines.append(f"{'✅' if o.correct else '❌'} [{kind}] 分數 {score}  第一名 {top}  正解 {expect}  {o.question.q}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="評估檢索品質")
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.yaml"))
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"設定錯誤：{exc}", file=sys.stderr)
        return 2
    store = VectorStore(settings.chroma_dir)
    if store.count() == 0:
        print("資料庫是空的，請先執行 docker compose run --rm ingest", file=sys.stderr)
        return 1
    llm = NvidiaClient(settings, min_interval=settings.ingest_min_interval)
    outcomes = [judge(q, retrieve(llm, store, settings, q.q)) for q in load_questions(args.questions)]
    print(
        f"設定：chunk={settings.chunk_strategy}，retrieve_k={settings.retrieve_k}，"
        f"rerank_k={settings.rerank_k}，門檻={settings.rerank_threshold}"
    )
    print(summarize(outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 確認測試通過**

Run: `docker compose run --rm test tests/test_eval.py -v`
Expected: 6 passed

- [ ] **Step 5: 起草題庫 `eval/questions.yaml`**

執行者用 Read 工具讀完 `data/ocr/living-room-ac/` 所有 `.md`（必要時對照 `data/pages/` 圖片），寫出 15 題：
- 6 題「一般問題」：用到原文關鍵字（例如功能名稱、按鍵名稱）
- 6 題「換句話說」：家人口語、不用原文關鍵字（例如「想讓房間快點變涼」對應「強力運轉」）
- 3 題「說明書沒有」：`expect: []`（例如「冷氣可以用來烘衣服嗎？」、「冷媒要多久補一次？」需確認說明書真的沒寫）
- 每題的 `pdf_page` 必須是讀過該頁 OCR 內容確認的 PDF 頁序（檔名 `pNN` 的數字），**不是**印刷頁碼

格式：
```yaml
# 正解用 PDF 頁序（data/pages 檔名 pNN 的數字），不是說明書上印的頁碼
- q: 想讓房間快點變涼要按哪個鍵？
  expect:
    - {doc_id: living-room-ac, pdf_page: 11}
- q: 冷氣可以用來烘衣服嗎？
  expect: []
```

- [ ] **Step 6: 👤 使用者步驟：修改題目說法**

請使用者看過 `eval/questions.yaml`，把題目改成家人實際會問的說法，有想到的問題可以加進去（記得標正解頁）。

- [ ] **Step 7: 執行 eval 並調整門檻**

Run: `docker compose run --rm app python -m eval.eval`
Expected: 印出設定、檢索命中率、拒答正確率、逐題分數

調門檻的方法：看逐題分數，找出「應拒」題目的最高分與「應答」題目（第一名正確者）的最低分，把門檻設在兩者之間。用 `.env` 加 `RERANK_THRESHOLD=<值>` 後重跑確認。

目標：檢索命中率 ≥ 80%、拒答正確率 3/3。未達標時把失敗題目與分數告訴使用者，一起決定是修 OCR 內容、改題目，還是試其他 chunk 策略（Step 8）。

確定後，把最後採用的門檻寫回 `rag/config.py` 的 `rerank_threshold` 預設值，`.env` 的 `RERANK_THRESHOLD` 刪除；並把 `EXAMPLE_QUESTIONS` 換成題庫中 4 題最常見、且 eval 答對的「一般問題」。

- [ ] **Step 8（學習實驗，選做）：比較 chunk 策略**

```powershell
docker compose run --rm -e CHUNK_STRATEGY=page ingest
docker compose run --rm -e CHUNK_STRATEGY=page app python -m eval.eval
docker compose run --rm -e CHUNK_STRATEGY=fixed ingest
docker compose run --rm -e CHUNK_STRATEGY=fixed app python -m eval.eval
docker compose run --rm ingest
```

（OCR 已有結果，只會重做 chunk 與 embedding。最後一行換回預設的 heading 策略。）把三種策略的命中率整理給使用者看。

- [ ] **Step 9: 跑全部測試並 Commit**

Run: `docker compose run --rm test`
Expected: 全部 passed

```powershell
git add eval tests/test_eval.py rag/config.py
git commit -m "feat: add retrieval eval with question set and tuned threshold" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 前面所有指令
- Produces: 使用說明（給使用者自己，以及日後想參考這個專案的人）

- [ ] **Step 1: 寫 `README.md`**

````markdown
# 家電說明書小幫手（掃描版說明書 RAG）

把掃描版家電說明書 PDF 變成可以用手機問問題的網頁：回答會附上說明書原頁圖片。
手寫的 RAG 管線（不用 LangChain / LlamaIndex），每個步驟一個檔案，方便學習與調整。

## 運作方式

```
建索引（ingest，事先跑一次）
  PDF → 頁面圖片（rag/pdf.py）→ 視覺模型 OCR（rag/ocr.py）→ 切 chunk（rag/chunk.py）
      → embedding → 存入 Chroma（rag/store.py）

問答（app，每次有人問）
  追問改寫 → 向量檢索 top 10 → rerank 留 top 3 → 門檻判斷（rag/retrieve.py）
      → LLM 串流回答 → 找出引用頁面圖（rag/answer.py、rag/service.py）
```

模型使用 NVIDIA NIM 免費 API（build.nvidia.com），所有呼叫集中在 `rag/llm.py`。

## 第一次設定

1. 安裝 Docker Desktop
2. 到 https://build.nvidia.com 登入，任一模型頁按「Get API Key」
3. `Copy-Item .env.example .env`，填入 `NVIDIA_API_KEY`
4. 把說明書 PDF 放進 `manuals/`，在 `manuals/manuals.yaml` 登記：
   ```yaml
   - file: 客廳冷氣.pdf
     doc_id: living-room-ac   # 小寫英文、數字、-
     name: 客廳冷氣
   ```
5. 確認模型可用：`docker compose run --rm app python scripts/probe_models.py`

## 常用指令

| 做什麼 | 指令 |
|---|---|
| 建索引（新增說明書、修正 OCR 後） | `docker compose run --rm ingest` |
| 只處理某一本 | `docker compose run --rm ingest --doc living-room-ac` |
| 全部重新 OCR（會覆蓋人工修正） | `docker compose run --rm ingest --force` |
| 啟動問答網頁 | `docker compose up app` |
| 改了程式碼後重啟 | `docker compose up --build app` |
| 跑測試 | `docker compose run --rm test` |
| 評估檢索品質 | `docker compose run --rm app python -m eval.eval` |

## 用手機連線

電腦執行 `ipconfig` 找 Wi-Fi 的 IPv4 位址，手機（同一個 Wi-Fi）開 `http://<那個IP>:7860`。
連不上時檢查 Windows 防火牆是否允許 Docker Desktop 使用私人網路。

## 修正 OCR

OCR 結果在 `data/ocr/<doc_id>/pNN.md`（`NN` 是 PDF 頁序），可以直接編輯。
重跑 `docker compose run --rm ingest` 時已存在的檔案不會重新 OCR，只重做 chunk 與 embedding。

## 調整

所有設定在 `rag/config.py`，可用 `.env` 覆寫（見 `.env.example`）：模型型號、chunk 策略（heading / page / fixed）、檢索數量、rerank 門檻。
改了 chunk 策略要重跑 ingest；改完用 eval 比較命中率。

## Docker 小抄

- **image**：裝好 Python、套件、程式碼的快照；`app`、`ingest`、`test` 三個 service 共用同一個 image，只差啟動指令
- **bind mount**：`manuals/`、`data/`、`eval/` 從電腦資料夾掛進 container，container 刪掉資料仍在
- **port**：`7860:7860` 是「電腦 port:container port」，電腦的 7860 被佔用時改左邊即可
- **.env**：API key 只在執行時以環境變數傳入，不會打包進 image

## 注意

- 說明書 PDF 版權屬原廠，`*.pdf` 已排除在 git 之外；`data/` 也不進 git
- NVIDIA 免費 API 屬開發用途、約每分鐘 40 次請求，條款可能變動；換供應商只需修改 `rag/llm.py`
````

- [ ] **Step 2: 最終驗證**

Run: `docker compose run --rm test`
Expected: 全部 passed

Run: `git status --short --ignored`
Expected: `.env`、`data/`、`manuals/客廳冷氣.pdf` 都在 `!!`（已忽略）清單中，沒有出現在待 commit 檔案裡

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m "docs: add README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
