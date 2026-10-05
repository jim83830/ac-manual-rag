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

## 設計取捨與踩過的坑

**為什麼不用 LangChain / LlamaIndex？**
這個專案的目的是弄懂 RAG 每一步在做什麼。每個步驟一個小模組（`pdf → ocr → chunk → store → retrieve → answer`），出問題時能直接定位在哪一步，也能單獨替換或調參。而且掃描版 PDF + 視覺 OCR 本來就不是框架的標準流程，用框架還是得自己寫客製元件。

**為什麼用視覺模型（VLM）做 OCR，而不是傳統 OCR？**
這本說明書是掃描檔，沒有文字層，內容又大量依賴遙控器按鍵圖示和表格。傳統 OCR 只會「認字」：圓圈裡的「快速」會變成普通的兩個字，表格會散成零碎的行。VLM 可以照 prompt 的規則整理版面：按鍵一律寫成【快速】、表格轉 Markdown、圖片寫成 `[圖：…]`。統一的【】格式也讓「快速鍵是幹嘛的」這種問題能精準檢索到。

**chunk 怎麼切？**
說明書一頁通常有好幾個獨立主題，所以預設依 `##` 標題切，每段剛好是一個完整的操作說明；太長才再依段落切。每個 chunk 前面加上脈絡前綴（`客廳冷氣 > 運轉的方式 > 強力運轉`），讓 embedding 知道這段在講什麼。策略可以切換成整頁或固定字數，用 eval 比較。

**為什麼要 rerank？效果如何？**
向量檢索快但粗，reranker 把「問題 + 段落」放在一起逐一比對，準很多但慢，所以只對向量檢索的前 10 名做。實際例子：問「想讓房間快點變涼要怎麼做？」，最相關的「強力運轉的內容」在向量檢索只排第 6，rerank 後變第 1。

**怎麼避免 LLM 亂編？**
兩道防線：rerank 最高分低於門檻就直接回「說明書裡找不到」，不呼叫 LLM；通過門檻後，system prompt 規定只能根據提供的段落回答、沒寫就說沒寫。目前門檻還沒用完整題庫調好，「冷氣可以烘衣服嗎？」這類問題實際上是靠第二道防線擋下的，下一步是用 eval 題庫找出能分開「可答 / 不可答」的門檻。

**頁碼為什麼要存兩個？**
PDF 的頁序和說明書上印的頁碼不同（前面有封面、空白頁，PDF 第 11 頁印的是「9」）。程式用 `pdf_page` 找圖片，回答時引用 `printed_page`，家人拿紙本對照才找得到。

**追問怎麼處理？**
「那要怎麼取消？」單獨拿去檢索什麼都找不到。有對話紀錄時，先請 LLM 把問題改寫成完整的「強力運轉要怎麼取消？」再檢索（query rewriting）。

**免費 API 遇到什麼問題？**
- 模型會無預警下架（HTTP 410），原本計畫用的聊天、embedding、rerank 模型開工時都已下架 → 用探測腳本實際呼叫確認，模型名稱全部放設定檔
- 端點會整個塞住（請求掛到逾時），但帳號本身沒事 → 用假 key（很快回 403）和其他模型做對照，才分得出是平台問題還是帳號被限流
- 所以 OCR 可以獨立換平台（OpenAI 相容格式，只換網址和 key），最後改用 Google Gemini 的免費方案

**遇過最難抓的 bug？**
推理型模型的「思考」也算在 `max_tokens` 裡，OCR 輸出被截斷卻沒有任何錯誤，存下來的檔案看起來很正常，有一頁只轉了 24 項中的 4 項。修正：檢查 `finish_reason == "length"` 就視為失敗、不存檔，並提高上限；再用「OCR 字數 ÷ 頁面墨色比例」篩出內容異常少的頁面。

**重試策略為什麼 OCR 和問答不一樣？**
Gemini 免費方案每天只有 20 次請求，SDK 預設的自動重試讓 2 頁失敗就燒掉 12 次額度。所以 OCR 預設不重試，失敗的頁面留給下次 ingest 補做（本來就會跳過已完成的頁面）；問答的呼叫沒有每日額度問題，保留重試，避免家人遇到暫時性錯誤。

**ingest 中途中斷會怎樣？**
`.md` 是正本、資料庫是從它算出來的副本，任何時候中斷都能重跑恢復。OCR 結果先寫暫存檔再改名（原子寫入），不會留下寫一半、下次又被當成完成的檔案；資料庫要等 embedding 成功後才替換。

**OCR 為什麼用多執行緒？不會被 GIL 卡住嗎？**
一頁 OCR 的時間幾乎都在等網路回應，等 I/O 時執行緒會放掉 GIL，所以多頁能同時等。要注意的是限流函式「讀上次時間 → 等待 → 更新」不是原子操作，要另外加鎖。測試用 `threading.Barrier` 證明真的有並行：兩個呼叫必須同時進行，柵欄才會放行。

**怎麼測試？**
單元測試全部用假模型（`FakeLLM`、`httpx.MockTransport`），不連網、結果固定；真實 API 用探測腳本另外檢查。檢索品質用 eval 題庫量化：命中率（正解頁有沒有在前 3 名）和拒答正確率。

**已知限制與下一步**
- 曲線圖只會被描述成 `[圖：…]`，圖裡的數值變化需要人工補成文字
- 跨頁主題會被切成兩個 chunk
- 待做：用 eval 題庫調門檻、比較不同 chunk 策略、部署到 Hugging Face Spaces（電腦不用一直開著）

## Docker 小抄

- **image**：裝好 Python、套件、程式碼的快照；`app`、`ingest`、`test` 三個 service 共用同一個 image，只差啟動指令
- **bind mount**：`manuals/`、`data/`、`eval/` 從電腦資料夾掛進 container，container 刪掉資料仍在
- **port**：`7860:7860` 是「電腦 port:container port」，電腦的 7860 被佔用時改左邊即可
- **.env**：API key 只在執行時以環境變數傳入，不會打包進 image

## 注意

- 說明書 PDF 版權屬原廠，`*.pdf` 已排除在 git 之外；`data/` 也不進 git
- NVIDIA 免費 API 屬開發用途、約每分鐘 40 次請求、回應有時很慢（視覺 OCR 一頁可能 3～4 分鐘），條款與模型清單可能變動；換供應商只需修改 `rag/llm.py`
