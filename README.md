# AI 職安稽核系統（LINE Bot）

使用者在 LINE 傳送工地照片，系統用 YOLO 偵測人員和防護裝備，再交給視覺 LLM 對照官方稽核項次和法規，產生稽核結果，最後推送回 LINE。

```
LINE ──webhook──▶ Cloudflare Tunnel ──▶ app.py (Flask, :5443)
                                          │
                                          ├─ YOLO（best.pt，大圖用 SAHI 切片）
                                          └─ safety_audit_system.py
                                                 │  OpenAI 相容 API
                                                 ▼
                         ┌───────────────────────┴───────────────────────┐
                 Ollama (:11434)                              llm_server.py (:8001)
                 gemma4:e4b-it-qat                            Gemma 3n E2B + 稽核 LoRA
```

LLM 後端由 `.env` 決定，有兩種：

| 後端 | 模型 | 啟動方式 | 速度（實測，RTX 5060 8GB） |
|---|---|---|---|
| Ollama | `gemma4:e4b-it-qat` | Ollama 常駐即可 | 較快 |
| LoRA 推論服務 | `google/gemma-3n-E2B-it` + `gemma3n-e2b-audit-lora/final` | `python llm_server.py` | 約 4–7 tok/s，一張圖約 1.5–2.5 分鐘 |

---

## 環境

所有指令都在 conda 的 **`safety-bot`** 環境執行。這個環境裝的是 torch 2.11 + CUDA 12.8，RTX 50 系列顯卡需要 cu128。

```powershell
conda activate safety-bot
pip install -r requirements.txt         # 主程式：Flask、LINE SDK、YOLO 等
pip install -r requirements-lora.txt    # 只有要用 LoRA 後端時才需要
```

`.env` 的必要設定，參考 `.env.example`：

- `CHANNEL_ACCESS_TOKEN`、`CHANNEL_SECRET`：LINE Messaging API 的憑證。
- `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`LLM_SUPPORTS_VISION`：LLM 後端，設定方式見下一節。
- `DATABASE_URL`：選填。沒有設定時不會寫入 PostgreSQL。

---

## 切換 LLM 後端

修改 `.env`，兩組設定只保留一組不加註解，改完要重新啟動 `app.py`。

**Ollama gemma4：**
```ini
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=gemma4:e4b-it-qat
LLM_SUPPORTS_VISION=true
```

**Gemma 3n 稽核 LoRA：**
```ini
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=local
LLM_MODEL=gemma3n-e2b-audit-lora
LLM_SUPPORTS_VISION=true
```

> 8GB 顯卡同一時間只能跑一個 LLM。改用 LoRA 之前，先執行 `ollama ps`，確認沒有載入中的模型；有的話用 `ollama stop <模型名稱>` 釋放 VRAM。

---

## 啟動 LINE Bot

### 1. 使用 LoRA 時，先啟動推論服務（第一個視窗）

```powershell
conda activate safety-bot
python llm_server.py
```

- 第一次執行前要先完成兩件事：
  1. 到 <https://huggingface.co/google/gemma-3n-E2B-it> 同意 Gemma 授權條款。
  2. 執行 `hf auth login` 登入。如果瀏覽器沒有自動開啟，就改選 *Paste an access token*，貼上在 <https://huggingface.co/settings/tokens> 建立的 Read token。
- 第一次會下載 base model，大約 10GB，存在 `%USERPROFILE%\.cache\huggingface`。
- 看到 `Uvicorn running on http://127.0.0.1:8001` 就代表啟動完成。可以用 `http://127.0.0.1:8001/health` 確認狀態。

### 2. 啟動主程式（第二個視窗）

```powershell
conda activate safety-bot
python app.py
```

| 參數 | 說明 |
|---|---|
| `--port 5443` | 監聽的 port，預設是 5443 |
| `--http` | 強制用 HTTP。`certs/dev-cert.pem` 不存在時也會自動改用 HTTP |
| `--no-tunnel` | 不啟動 Cloudflare Tunnel |

### 3. 更新 LINE Webhook

Cloudflare 免費 tunnel **每次啟動都會換一個網址**。啟動後在 log 找這一行：

```
LINE Webhook        : https://xxxx-xxxx.trycloudflare.com/line/webhook
```

到 LINE Developers Console → 你的 channel → **Messaging API** → **Webhook URL**，貼上網址後按 **Verify**，並確認 **Use webhook** 有開啟。

用瀏覽器打開這個網址，看到 `{"ok": true, ...}` 就代表 tunnel 是通的。

### 4. 測試

在 LINE 傳一張工地照片。bot 會先回「已收到照片」，分析完成後再推送結果。用 LoRA 後端時，大約要等 1.5–2.5 分鐘。

---

## LoRA 推論服務（`llm_server.py`）

這是一個以 transformers 載入 `google/gemma-3n-E2B-it`、並套用 `gemma3n-e2b-audit-lora/final` 的 OpenAI 相容服務。

**端點：**
- `POST /v1/chat/completions`：支援文字和圖片（`image_url` 可以是 data URL 或 http(s) 網址）、`max_tokens`、`temperature`、`top_p`、`response_format: json_object`（會去掉 ```json 標記），以及 `stream: true`。
- `GET /v1/models`、`GET /health`。
- 不支援 tools / function calling，也不支援音訊。Ollama 專用的參數（`options`、`think`）會被忽略。

**環境變數**（都可以不設，可以寫在 `.env`）：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `LORA_BASE_MODEL` | `google/gemma-3n-E2B-it` | base model |
| `LORA_ADAPTER_DIR` | `./gemma3n-e2b-audit-lora/final` | LoRA adapter 目錄 |
| `LORA_OFFLOAD_PLE` | `1` | 把 per-layer embedding 放在 CPU |
| `LORA_HOST` / `LORA_PORT` | `127.0.0.1` / `8001` | 監聽位址 |
| `LORA_MODEL_NAME` | `gemma3n-e2b-audit-lora` | 對外回報的模型名稱 |

**記憶體配置**（針對 8GB 顯卡設計）：

- 啟動時會把 LoRA 合併進權重，整個模型以 **bf16** 執行，不做量化。
- Gemma 3n 的 per-layer embedding 約有 20 億參數、佔 4GB，留在 **CPU**。它每個 token 只需要查表一次，實測每個 token 只多花約 0.13ms。
- 音訊編碼器（約 7 億參數）用不到，直接移除。
- 載入後 GPU 佔用約 5.1GB。稽核 prompt 約 1 萬 token 時，峰值約 7.3GB。
- 系統記憶體約需 5GB（per-layer embedding 加上 Python 本身）。

**每次請求的 log：**
```
[LoRA] prompt=11786 completion=571 tokens，耗時 149.3s (3.8 tok/s)
```

---

## 已知限制

- **LoRA 後端比較慢。** 解碼速度約 4–7 tok/s，瓶頸在 transformers 逐 token 發出大量小的 CUDA 運算，GPU 本身不是瓶頸。常見的加速方法（`torch.compile` 加 CUDA graphs）需要 Triton，而 Windows 版 PyTorch 沒有附 Triton。
- **prompt 太長會變慢。** 超過約 1.5 萬 token 時，VRAM 會溢出到系統共享記憶體。程式不會出錯，但會明顯變慢。目前的稽核 prompt 約 1.0–1.2 萬 token。
- **LoRA 判定品質還沒有系統化比較。** 曾經發生 YOLO 偵測到 `no_boots`，但 LoRA 仍然判定「人員均穿著安全鞋」、相關項目全部通過的情況。正式採用前，建議先用 `test_images/` 和 gemma4 做對照。
- **同一時間只處理一個請求。** 多張照片同時送進來時會排隊。

---

## 疑難排解

| 現象 | 原因和處理方式 |
|---|---|
| LINE 沒有任何回應 | Webhook 網址還是舊的。看 `app.py` 的 log，把新網址貼到 LINE 後台並按 Verify |
| 稽核結果顯示「LLM 呼叫失敗：Connection error」 | `.env` 指向 LoRA，但 `llm_server.py` 沒有啟動 |
| `llm_server.py` 出現 `CUDA out of memory` | Ollama 還佔著 VRAM：先 `ollama ps`，再 `ollama stop <模型名稱>` |
| 下載時出現 401 / 403（gated repo） | 還沒同意 Gemma 授權條款，或還沒執行 `hf auth login` |
| 出現 `RequestsDependencyWarning` | 只是套件版本警告，不影響執行 |
| cloudflared 出現 `ERR Cannot determine default origin certificate path` | quick tunnel 的正常訊息，可以忽略 |

---

## 主要檔案

| 檔案 | 說明 |
|---|---|
| `app.py` | Flask 主程式：LINE webhook、Cloudflare Tunnel、報表匯出 |
| `safety_audit_system.py` | YOLO 偵測、組合稽核 prompt、呼叫 LLM、解析 JSON |
| `llm_server.py` | Gemma 3n 加 LoRA 的 OpenAI 相容推論服務 |
| `gemma3n-e2b-audit-lora/final/` | 稽核用的 LoRA adapter（PEFT） |
| `benchmark.py`、`make_report.py` | LLM 效能測試和報告產生 |
| `regulations/` | 法規知識庫 |
| `test_images/` | 各類別的測試照片 |
| `requirements.txt`、`requirements-lora.txt` | 主程式和 LoRA 服務各自需要的套件 |
