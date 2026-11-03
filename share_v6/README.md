# v6 微調資料、設定與評測（給晴茵）

這包是我們 v6 微調用的資料、腳本、評測結果，另外附上妳 test_rag.jsonl 和我們資料的重疊比對。
模型權重另外一包（`v6_model.zip`）。

## 先講最重要的兩件事

**1. 我們 v6 的分數偏高，不能直接拿來跟妳比。**
我們的 test_100 是從全部標註隨機抽 100 張，沒排除訓練集。事後檢查，100 張裡有 91 張是 v6 訓練過的，真正沒看過的只有 9 張。
所以「準確率 78%、精確度 81%、召回率 63%」大部分是在訓練圖上量的。

**2. 妳的 test_rag 有 93 張在我們的 v6 訓練集裡。**
妳的 157 張中，「內部」100 張和我們用同一批照片：93 張在我們 train、6 張在我們 val、1 張都不在。
「外部」57 張（kcg_hazard 系列）我們沒有。
所以如果拿我們的 v6 模型跑妳的測試集，內部那 100 張會偏高，只有 57 張外部加 6 張 val 比較公平。
逐張對照見 `overlap_test_rag.csv`。

## v6 訓練設定

| 項目 | 設定 |
|---|---|
| 底模型 | google/gemma-3n-E2B-it |
| 量化 | QLoRA 4-bit：nf4、double quant、compute bf16 |
| LoRA | r=32、alpha=32、dropout=0.05，target = q/k/v/o/gate/up/down_proj（可訓練參數 45.8M，0.84%） |
| Epoch | 3 |
| 學習率 | 5e-5，warmup_ratio 0.05 |
| Batch | 1 × gradient accumulation 8 |
| Optimizer | paged_adamw_8bit，bf16，gradient checkpointing |
| 圖片 | 長邊縮到 896px |
| Loss | 只在 assistant 的答案上算；system prompt、YOLO 清單、圖片 token 全設 -100 |
| 訓練時間 | RTX 3060 12GB 約 5.1 小時，最後 train loss 0.61 |

腳本：`scripts/finetune_gemma3n.py`（檔名沒有 v6，但 OUT_DIR 是 v6）。

## v6 資料

- 來源：`答案.xlsx` + 同一批 UploadImg 照片，共 483 張可用，切成 train 426 張、val 57 張（random seed 42）。
- train 裡「有缺失」的樣本複製 2 份，所以 train.jsonl 是 620 筆。
- 每筆格式：
  - `image`：圖片路徑（已改成 `UploadImg/<檔名>`，妳那邊路徑是 `UploadImg/UploadImg/<檔名>`，對一下就好）
  - `prompt`：`請分析這張施工現場照片的職安合規性。` + 空一行 + YOLO 偵測清單
  - `response`：`{"compliant": true/false, "findings": [{"item": "...", "result": "..."}]}`，item/result 直接取自答案.xlsx 原文，合規時 findings 為空陣列
- system prompt：
  > 你是專業的工地職安稽核員。分析照片後只輸出 JSON：{"compliant": true/false, "findings": [{"item":"項目名稱","result":"缺失描述"}]}，符合規定時 findings 為空陣列。
- YOLO 清單是我們自己的 best.pt 跑的（16 類，PPE＋machinery/vehicle/safety_cone），格式見 `scripts/benchmark.py` 的 `run_yolo`。

## 評測方式

- 腳本：`scripts/test_100_real_flow_v6.py`
- 指標是**整張圖二分類**：模型輸出的 findings 非空就算「預測違規」；正類 = 違規。
- 解碼：greedy（do_sample=False），max_new_tokens=512。
- JSON 解析失敗算 PARSE_ERROR，**不計入**準確率、精確度、召回率的分母（v6 有 3 張）。
- 不比對項次文字，只看有沒有違規。

| 版本 | 準確率 | 精確度 | 召回率 | 解析失敗 |
|---|---|---|---|---|
| v5 | 75.8% | 80.0% | 52.6% | 5 |
| v6 | 78.4% | 81.3% | 63.4% | 3 |

（再提醒一次：這兩行都有訓練集重疊的問題。）

## 我們跟妳的差異（可能影響分數）

1. **任務不一樣**：我們只輸出有沒有違規和缺失描述；妳的要輸出「項次代碼＋結果＋缺失條列」，還有 RAG 候選條文。如果妳的指標要求項次也對，分數本來就會比二分類低很多。
2. **妳的準確率 22.9% 比「全部猜合規」還低**（妳的測試集 95/157 合規，全猜合規就有 60.5%）。這通常不只是模型沒學好，建議先檢查：
   - 正類有沒有設反（違規 vs 合規）
   - 解析失敗、空輸出、格式跑掉的是不是被算成錯
   - 是不是逐項比對項次文字
   - 推論時的 system prompt、YOLO 清單、RAG 候選、chat template 跟訓練時是否一致
   - loss 是不是只算答案部分
3. 方便的話傳幾筆模型原始輸出給我們看，比較好判斷。

## 檔案

```
README.md                      這份說明
overlap_test_rag.csv           妳的 157 張 vs 我們 train/val 的逐張對照
data/train.jsonl               v6 訓練資料（620 筆）
data/val.jsonl                 v6 驗證資料（57 筆）
data/答案.xlsx                  原始標註
results/test_100_v6_result.json  v6 逐張評測結果（含模型原始輸出）
results/ablation_summary.json  各版本評測摘要
scripts/finetune_gemma3n.py    v6 訓練腳本
scripts/build_finetune_dataset.py  產生 train/val.jsonl
scripts/test_100_real_flow_v6.py   v6 評測腳本
scripts/eval_answer_key.py     讀答案.xlsx（評測用）
scripts/benchmark.py           run_yolo（產生 YOLO 清單）
```

`v6_model.zip`（另一包）：
```
gemma4-e2b-audit-lora-v6/final/   v6 LoRA adapter（搭配 google/gemma-3n-E2B-it）
best.pt                           我們的 YOLO 權重
```

腳本裡的路徑是我們本機的資料夾結構（uploads/UploadImg、答案.xlsx 放在腳本同層），要跑的話改一下 IMG_DIR 和路徑。金鑰類的設定都從 .env 讀，這包沒附 .env。
