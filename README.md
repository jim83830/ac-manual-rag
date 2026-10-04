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
目前預設的模型寫在 `rag/config.py`；NVIDIA 的免費模型會不定期下架，遇到 410 錯誤時用探測腳本找替代品。

## 第一次設定

1. 安裝 Docker Desktop
2. 到 https://build.nvidia.com 登入，任一模型頁按「Get API Key」（一把 key 可用所有模型）
3. `Copy-Item .env.example .env`，填入 `NVIDIA_API_KEY`
4. 把說明書 PDF 放進 `manuals/`，在 `manuals/manuals.yaml` 登記：
   ```yaml
   - file: 客廳冷氣.pdf
     doc_id: living-room-ac   # 小寫英文、數字、-
     name: 客廳冷氣
   ```
5. 確認模型可用：`docker compose run --rm app python scripts/probe_models.py`

OCR（視覺模型）可以單獨改用其他 OpenAI 相容平台，例如 Google Gemini：到 https://aistudio.google.com 取得免費 API key，
在 `.env` 設定 `VISION_BASE_URL`、`VISION_API_KEY`、`VISION_MODEL`（範例見 `.env.example`）。聊天、embedding、rerank 仍走 NVIDIA。

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
| 檢查模型是否可用 | `docker compose run --rm app python scripts/probe_models.py` |

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
- NVIDIA 免費 API 屬開發用途、約每分鐘 40 次請求、回應有時很慢（視覺 OCR 一頁可能 3～4 分鐘），條款與模型清單可能變動；換供應商只需修改 `rag/llm.py`
