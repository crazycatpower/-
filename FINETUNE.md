# 稽核 LoRA 微調（Gemma 3n E2B）

VLM 看工地照片判斷「有缺失／符合規定」並列出缺失項目。PPE 裝備主要由 YOLO（`best.pt`）負責，VLM 主攻違規操作與設施缺失。

## 流程

```powershell
python build_finetune_dataset_v10.py     # 答案.xlsx + 網路照片 → finetune_data_v10/{train,dev}.jsonl
python finetune_gemma3n_v10.py           # → gemma3n-e2b-audit-lora-v10/{best,last}；中斷後重跑會接著訓練
python test_val57.py gemma3n-e2b-audit-lora-v10/best                 # 內部 57 張
python test_val57.py gemma3n-e2b-audit-lora-v10/best --set ext113    # 外部 113 張
python test_val57.py base                                            # 底座 zero-shot 對照
```

三支腳本共用 `audit_vlm_common.py`（模型載入、prompt、YOLO 偵測清單、輸出解析）。評測結果寫在 `results/`。

| 資料 | 張數 | 用途 |
|---|---|---|
| `finetune_data_v10/train.jsonl` | 504（內部 379、網路 125），每張 3 個版本 | 訓練 |
| `finetune_data_v10/dev.jsonl` | 48（內部） | 訓練中挑 checkpoint |
| `finetune_data/val.jsonl`（val57） | 57（內部） | 測試，歷史版本共用 |
| `新增資料集/`（ext113） | 113（勞檢單位手冊） | 測試，不進訓練 |

## v1~v9 為什麼卡關

Gemma 3n 的第 20~29 層共用第 18、19 層的 key/value。transformers 5.0.0.dev0 只有 `use_cache=True` 才做這個共用；`generate()` 有開 cache，但 TRL 的 SFTTrainer 和 gradient checkpointing 訓練時會關掉，第 20~29 層就改用自己那組沒被用過的投影，等於換了一個模型。

4-bit 底座、24 張訓練圖的 loss：

| | 推論的 forward（有 cache） | 訓練的 forward（無 cache） |
|---|---|---|
| 不掛 adapter | 2.02 | 6.25 |
| 掛 v6 adapter | 1.20 | 0.04 |

v6 在訓練用的 forward 上把訓練集背起來，換回推論幾乎沒效果，所以加資料、調 epoch 都沒用。4-bit 量化不是原因。`audit_vlm_common.patch_kv_sharing` 讓沒有 cache 的 forward 也共用 KV；v10 的訓練與評測都經過它。

## 成績

嚴格準確率把輸出壞掉（JSON 解析失敗）算成判錯；方括號是 95% 信賴區間。

**val57（57 張）**

| 模型 | 嚴格準確率 | precision | recall | F1 | 輸出壞掉 |
|---|---|---|---|---|---|
| 底座，沒微調 | 0.649 [0.52, 0.76] | 0.647 | 0.524 | 0.579 | 4 |
| v6 | 0.719 [0.59, 0.82] | 0.696 | 0.727 | 0.711 | 3 |
| v10 `best`（第 200 步，2.2 epoch） | 0.842 [0.73, 0.92] | 0.895 | 0.708 | 0.791 | 0 |
| v10 第 60 步（0.65 epoch） | 0.842 [0.73, 0.92] | 0.857 | 0.750 | 0.800 | 0 |
| v10 第 140 步（1.5 epoch） | 0.754 [0.63, 0.85] | 0.667 | 0.833 | 0.741 | 0 |

**ext113（113 張）**

| 模型 | 嚴格準確率 | precision | recall | F1 | 輸出壞掉 |
|---|---|---|---|---|---|
| 底座，沒微調 | 0.566 [0.47, 0.65] | 0.474 | 0.837 | 0.605 | 2 |
| v6 | 0.602 [0.51, 0.69] | 0.500 | 0.930 | 0.650 | 2 |
| v10 `best`（第 200 步） | 0.690 [0.60, 0.77] | 0.563 | 0.837 | 0.673 | 0 |
| v10 第 60 步 | 0.646 [0.55, 0.73] | 0.521 | 0.884 | 0.655 | 0 |
| v10 第 140 步 | 0.593 [0.50, 0.68] | 0.482 | 0.954 | 0.641 | 0 |

- `best` 是用 dev 挑的（共訓練 368 步、4 個 epoch；第 200 步的 dev 判定準確率 0.833 最高），val57 與 ext113 都沒有參與挑選。第 60、140 步是訓練中途留下的對照，第 60 步的 adapter 在 commit `94815a5` 裡。
- 各 checkpoint 之間起伏很大：第 140 步在兩個測試集都明顯較差，dev 準確率也在 0.71~0.83 之間跳動。第 200 步之後訓練 loss 降到 0.03 而 dev loss 回升，是過擬合。
- ext113 上 v10 與 v6 的信賴區間仍然重疊，v10 的主要問題是誤報（`best` 在合規 70 張裡判錯 28 張）。外部照片多是手冊裡的示意圖，和內部稽核照片差異大。

歷史紀錄（舊評測程式，val57；腳本與原始結果在 git 歷史裡，`6c14489` 之前）：v6 0.737、v9 0.667、v9 checkpoint-120 0.719。v1~v5 用的是另一組抽樣（test_100，其中 91 張是訓練圖），數字不能和上表比。

## 已知限制

- 答案表的標註是「這張照片在某個稽核項目上是否合規」，不是「整張圖有沒有任何缺失」。只憑稽核項目、不看圖去猜，val57 就有 0.789。v10 沒有把稽核項目給模型；`finetune_gemma3n_v10.py --item`、`test_val57.py --item` 可以做這個實驗，還沒跑。
- 測試集只有 57 與 113 張，差距在 10 個百分點以內都可能是抽樣誤差。
- `llm_server.py` 預設仍載入 `gemma3n-e2b-audit-lora/final`（v1），而且實際稽核用的 prompt 和訓練 prompt 不同；換成 v10 之前要先在實際流程上驗證。設 `LORA_ADAPTER_DIR` 可切換。
- 訓練時 prompt 超過 2000 token 的樣本（YOLO 清單特別長，約 3%）會略過：顯存不足時 Windows 會溢到共享記憶體，不報錯但一步要一小時以上。
