# 冷氣說明書 RAG — 設計文件

> 狀態：**設計已確認，待使用者最後審閱**（2026-10-02）
> 下一步：審閱通過後撰寫實作計畫

## 0. 目標與關鍵決定

做一個家用的「家電說明書問答」小工具：家人用手機問問題，系統從說明書找出相關內容並回答，同時附上說明書原頁圖片。
**主要目的是學習 RAG**：完整手寫走過 OCR → chunk → embedding → 檢索 → rerank → 生成，每一步可單獨觀察與調整；家人使用為附帶。同時藉此熟悉 Docker。

| 項目 | 決定 | 原因 |
|---|---|---|
| 目的 | 學習 RAG 優先 | 想實際走過每一步 |
| 使用方式（MVP） | 本機 Docker 跑 Gradio 網頁，家中 Wi-Fi 下用手機連 | 最簡單；學習重點放在 RAG |
| 雲端部署 | **不在 MVP**，列為下一階段：推到 Hugging Face Spaces（Docker 部署），外面也能用、電腦可關機 | 本機版需電腦常開、限家中 Wi-Fi |
| Google Sites | 只能當入口頁，用 iframe 嵌入 HF Spaces 的 App（下一階段選配） | Google Sites 不能跑 Python |
| 模型供應商 | NVIDIA NIM（build.nvidia.com）免費 API，OpenAI 相容介面 | 免費；視覺、embedding、reranker、chat 都有 |
| 供應商抽象 | 所有 API 呼叫集中在 `rag/llm.py`；模型名稱放設定檔 | NIM 免費額度屬原型用途（約 40 次/分鐘，非正式公告），條款可能改，需可換 Gemini / Claude |
| 資料範圍 | MVP 只放 `客廳冷氣.pdf`；每個 chunk 帶 `doc_id` 與頁碼 metadata | 之後加說明書只要放 PDF、登記、重跑 ingest |
| 回答呈現 | 回答 + 來源頁面掃描圖 | 說明書大量依賴遙控器按鍵圖示 |
| 實作方式 | 手寫管線 + 輕量套件（`openai` SDK、`chromadb`、`gradio`、`pypdfium2`），**不用** LangChain / LlamaIndex | 每一步看得懂、改得動 |
| 評估 | 測試題庫（約 15 題），量化檢索命中率與拒答正確率 | 調參時有依據 |
| 打包 | Docker + docker compose | 使用者想熟悉 Docker；HF Spaces 支援 Docker 部署 |

具體模型型號（視覺、embedding、reranker、chat）在實作時到 build.nvidia.com 確認目前可用清單後填入設定檔；embedding 需支援中文，chat 優先選中文表現好的模型。

## 資料來源觀察

- `manuals/客廳冷氣.pdf`：21 頁，**全部是掃描圖片，沒有文字層**；第 1 頁空白
- 繁體中文，大量遙控器按鍵圖示（如 `(快速)`、`[取消]`）、表格、示意圖 → 需用**視覺模型**轉寫
- **PDF 頁序 ≠ 印刷頁碼**：例如 PDF 第 11 頁底部印的是「9」（前有封面、空白頁）

## 1. 整體架構與 Docker

兩個階段，**同一個 image、兩個 compose service**（環境相同，只差啟動指令）：

- `ingest`（一次性，`docker compose run --rm ingest`）：PDF → 頁面 PNG → 視覺 OCR → chunk → embedding → 寫入 `data/`，做完即結束
- `app`（常駐，`docker compose up app`）：Gradio :7860，問題 → 改寫 → 檢索 → rerank → LLM → 回答 + 頁面圖

```
冷氣說明書RAG/
├── manuals/
│   ├── 客廳冷氣.pdf
│   └── manuals.yaml         # 說明書登記表
├── data/                    # ingest 產出（bind mount，不進 git）
│   ├── pages/<doc_id>/p11.png        # 檔名用 PDF 頁序
│   ├── ocr/<doc_id>/p11.md           # OCR 結果，可人工修正
│   └── chroma/                       # 向量資料庫
├── rag/                     # 每個檔案只做一件事
│   ├── config.py            # 模型名稱、chunk 策略、top-k、門檻、常見問題按鈕
│   ├── llm.py               # 包裝 NVIDIA API（chat / vision / embed / rerank）＋重試
│   ├── pdf.py               # PDF → PNG、空白頁偵測
│   ├── ocr.py               # 頁面圖 → Markdown
│   ├── chunk.py             # Markdown → chunks（多種策略）
│   ├── store.py             # Chroma 讀寫
│   ├── retrieve.py          # 問題改寫 + 向量檢索 + rerank + 門檻
│   └── answer.py            # 組 prompt、串流生成、解析引用頁碼
├── ingest.py                # ingest 入口（--force、--doc 參數）
├── app.py                   # Gradio 介面
├── eval/
│   ├── questions.yaml
│   └── eval.py
├── Dockerfile
├── docker-compose.yml
├── .env.example             # 範本；實際 .env 不進 git、不進 image
└── .gitignore               # 排除 .env、data/
```

Docker 重點：

- **Bind mount**：`manuals/`、`data/` 從主機資料夾掛進 container，container 刪掉資料仍在，也能在主機直接編輯 OCR 結果
- **Secrets**：`NVIDIA_API_KEY` 經 `.env` → 環境變數傳入，不打包進 image
- **Port**：container 7860 → 主機 7860，手機連 `http://<電腦區網IP>:7860`
- 下一階段部署 HF Spaces 時，另外 build 一個把 `data/` COPY 進去的 image（雲端沒有 bind mount），key 改用 Spaces Secrets

## 2. Ingest

0. **說明書登記表** `manuals/manuals.yaml`：
   ```yaml
   - file: 客廳冷氣.pdf
     doc_id: living-room-ac
     name: 客廳冷氣
   ```
1. **PDF → 頁面 PNG**：約 150 DPI，存 `data/pages/<doc_id>/pNN.png`（NN = PDF 頁序）；幾乎全白的頁面視為空白頁跳過
2. **視覺模型 OCR → Markdown**，統一格式：
   - 第一行 `頁碼：9`（讀頁面底部印刷頁碼；讀不到寫 `頁碼：無`）
   - 第二行 `分類：運轉的方式`（側邊分類標籤；沒有則省略）
   - 大標題 → `## 強力運轉`
   - 按鍵圖示 → `【快速】`、`【取消】`、`【時刻▽】`（統一用【】）
   - 表格 → Markdown 表格
   - 示意圖 → `[圖：簡短描述]`
   - 存 `data/ocr/<doc_id>/pNN.md`；**檔案已存在就跳過**（保留人工修正），`--force` 才重做
3. **切 chunk**：
   - 預設「依標題」：以 `##` 切段；單段超過約 800 字再依段落切
   - 每個 chunk 前加脈絡前綴：`客廳冷氣 > 運轉的方式 > 強力運轉`
   - 策略可在設定切換（`heading` / `page` / `fixed`），用 eval 比較
4. **Embedding → Chroma**：
   - metadata：`doc_id`、`doc_name`、`pdf_page`、`printed_page`（無則空）、`section`
   - 重跑時先刪除該 `doc_id` 的所有舊 chunk 再寫入（idempotent）

已知限制（MVP 接受）：跨頁主題會被切成兩個 chunk。

## 3. 檢索與回答

0. **問題改寫（支援追問）**：有對話紀錄時，請 LLM 把最新問題改寫成獨立完整的問題（「那要怎麼取消？」→「強力運轉要怎麼取消？」）；無對話紀錄則跳過
1. **向量檢索**：Chroma 取 top 10；函式保留 `doc_id` 過濾參數（MVP 介面不使用）
2. **Rerank**：NVIDIA reranker 重新評分，留 top 3
3. **門檻判斷**：最高 rerank 分數低於門檻 → 直接回「說明書裡找不到相關內容，建議聯絡原廠客服」，不呼叫 LLM。門檻值放設定檔，用 eval 的「說明書沒有」題目調整
4. **生成**（串流輸出），提供給 LLM 的每個段落標明說明書名與印刷頁碼。規則：
   - 只根據提供的段落回答，不知道就說不知道
   - 繁體中文；操作步驟用 1. 2. 3. 條列
   - 按鍵保留【】格式
   - 每個重點標註頁碼：有印刷頁碼寫「（第 9 頁）」，沒有則寫「（PDF 第 11 頁）」
   - 拆機、漏水、電線等問題提醒找專業人員
5. **附圖**：從回答中解析引用的頁碼，**只在本次 top 3 chunk 範圍內**對應回 `(doc_id, pdf_page)`，顯示對應掃描圖；解析不到時用 rerank 第一名的頁面

之後多本說明書：介面加「問哪一台？」下拉選單，以 `doc_id` 過濾檢索。

## 4. 介面（Gradio，手機優先）

- 標題「家電說明書小幫手」
- **常見問題按鈕**（點一下就送出，內容寫在設定檔）
- 對話區：回答下方顯示來源頁面縮圖，可點開放大
- **🔍 檢索細節**（每則回答下方，預設收合）：改寫後的問題、向量檢索 top 10、rerank 前後分數與排名——給開發者學習與調參用
- 輸入框 + 送出、**清除對話**按鈕（換話題時避免追問改寫被舊對話干擾）
- 存取保護：本機版不設密碼（僅區網可連）；HF Spaces 階段加 Gradio 帳號密碼

## 5. 測試題庫與 Eval

- `eval/questions.yaml`，約 15 題，**OCR 完成、讀過全部內容後才出題**；先由 Claude 起草，使用者改成家人實際會問的說法
- 三類題目：
  - 一般問題：用到原文關鍵字
  - 換句話說：不用原文關鍵字（考驗語意檢索）
  - 說明書沒有：應回「找不到」（約 3 題）
- 正解以 **`doc_id` + `pdf_page`** 標示（不用印刷頁碼，避免混淆）：
  ```yaml
  - q: 想讓房間快點變涼要按哪個鍵？
    expect: [{doc_id: living-room-ac, pdf_page: 11}]
  - q: 冷氣可以用來烘衣服嗎？
    expect: []
  ```
- 執行：`docker compose run --rm app python eval/eval.py`
- 輸出：目前策略設定、**檢索命中率**（正解頁出現在 rerank 後前 3 名的比例）、**拒答正確率**、逐題失敗清單
- 範圍：只評檢索與拒答，不評回答文字品質（MVP 不做 LLM 評審）

## 6. 錯誤處理

- **API 限流**：`llm.py` 統一處理，遇 429 / 暫時性錯誤以指數退避重試；ingest 呼叫間加小間隔
- **單頁 OCR 失敗**：記錄錯誤、跳過、繼續其他頁，結束時列出失敗頁；重跑只補做缺 `.md` 的頁面
- **問答時 API 失敗**：介面顯示「服務暫時無法使用，請稍後再試」，App 不中斷
- **未設定 API key**：啟動時檢查，直接報錯並提示設定 `.env`
- **Chroma 無資料**（尚未 ingest）：app 啟動時提示先執行 ingest

## 不在 MVP 範圍

- HF Spaces 部署、Google Sites 入口（下一階段）
- 介面上的說明書選單（第二本說明書加入時再做）
- 回答品質的自動評分（LLM-as-judge）
- 跨頁 chunk 合併
- 使用者帳號、對話紀錄保存
