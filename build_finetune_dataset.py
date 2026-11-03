# -*- coding: utf-8 -*-
"""
把 答案.xlsx + uploads/UploadImg 轉成微調用的 instruction/response 資料集。

答案.xlsx 的類別代號（1,6,7,9,...）跟 app 目前系統提示詞用的六大類
（墜落防止/倒塌崩塌防止/...）沒有對得上的官方對照表，硬套會生出錯誤標籤，
所以訓練目標改用答案本身的欄位（項目名稱、結果文字）直接當作目標輸出。

v2：把 item/result 合併成完整一句話的 finding 描述，並加上一句用真實
資料機械式生成（非模型自己編）的 summary，讓輸出更接近自然語言、更詳細。

v4：v2/v3 的 summary 是同一句固定模板，套用在近半數樣本上，查證這是造成
微調後模型生成時卡進重複迴圈的資料層面成因（模板化的合成回答提高重複機率）。

v5：對齊推論格式。查證發現訓練 prompt 是純文字、完全沒有 YOLO 偵測清單，
但評測腳本（test_100_real_flow.py）與未來的產品推論會在 prompt 裡塞一大段
YOLO 偵測框清單——模型訓練時從沒看過這種格式，推論時硬塞給它是 out-of-
distribution 輸入，實測讓 recall 從 0.56 掉到 0.41。v5 改成：
  1. 訓練時就用 benchmark.run_yolo 對每張圖跑真實 YOLO，把 detection_block
     接在 prompt 後面，格式與 test_100_real_flow.py 完全一致。
  2. prompt / system prompt / 輸出 schema 三者一次對齊 test_100_real_flow.py：
     user prompt = "請分析這張施工現場照片的職安合規性。\n\n{detection_block}"
     response    = {"compliant": bool, "findings": [{"item": "...", "result": "..."}]}
     （拿掉 v2~v4 的合成 summary；findings 改回結構化物件，內容取自答案原文）
  YOLO 偵測結果會快取到 finetune_data/yolo_cache.json，重跑不必再跑一次推論。

v6：v5 評測 recall 只有 0.53，漏抓集中在 PPE／墜落防護（未戴安全帽、未穿反光背心、
未張掛安全網、未設護欄）——模型對這些直接回 compliant=true。對安全稽核來說 false
negative（漏掉危害）比 false positive 嚴重，v5 precision 0.80 也還有餘裕，所以 v6
在 train split 裡把「有缺失」樣本複製 TRAIN_VIOLATION_OVERSAMPLE 份，讓模型多看危害
案例、把決策邊界往「敢開缺失」推。val split 不動，指標才可比。

輸出：finetune_data/train.jsonl, finetune_data/val.jsonl
每筆: {"image": "<絕對路徑>", "prompt": "...", "response": "<json字串>"}
"""
import json
import os
import random
from pathlib import Path

import openpyxl

from benchmark import run_yolo

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "uploads" / "UploadImg"
ANSWER_XLSX = BASE_DIR / "答案.xlsx"
OUT_DIR = BASE_DIR / "finetune_data"
YOLO_CACHE = OUT_DIR / "yolo_cache.json"

# 與 test_100_real_flow.py 的 user prompt 完全一致（後面接 YOLO detection_block）
PROMPT_PREFIX = "請分析這張施工現場照片的職安合規性。"

# v6：train split 裡「有缺失」樣本額外複製幾份（1 = 不變）。2 代表violation出現2次。
TRAIN_VIOLATION_OVERSAMPLE = 2


def load_per_image():
    wb = openpyxl.load_workbook(ANSWER_XLSX)
    ws = wb.worksheets[0]
    per_image = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        base = os.path.basename(str(row[0]).replace("\\", "/"))
        item_name = (row[3] or "").strip() if isinstance(row[3], str) else row[3]
        result = (row[4] or "").strip() if isinstance(row[4], str) else row[4]
        per_image.setdefault(base, []).append({"item": item_name, "result": result})
    return per_image


def load_yolo_cache():
    if YOLO_CACHE.exists():
        try:
            return json.loads(YOLO_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("[yolo-cache] 快取檔損毀，重新建立")
    return {}


def save_yolo_cache(cache):
    OUT_DIR.mkdir(exist_ok=True)
    YOLO_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def detection_block_for(img_name, img_path, cache):
    """回傳該圖的 YOLO detection_block 文字；優先吃快取，缺了才跑推論。"""
    if img_name in cache:
        return cache[img_name]
    _, block = run_yolo(img_path)
    cache[img_name] = block
    return block


def main():
    random.seed(42)
    per_image = load_per_image()
    cache = load_yolo_cache()

    examples = []
    missing = 0
    processed = 0
    for img, rows in per_image.items():
        img_path = IMG_DIR / img
        if not img_path.exists():
            missing += 1
            continue

        block = detection_block_for(img, img_path, cache)
        processed += 1
        if processed % 25 == 0:
            print(f"  ...已處理 {processed} 張，存一次快取")
            save_yolo_cache(cache)

        compliant = all(r["result"] == "符合規定" for r in rows)
        violations = [r for r in rows if r["result"] != "符合規定"]
        if compliant:
            findings = []
        else:
            findings = [
                {"item": str(r["item"] or "").strip(), "result": str(r["result"] or "").strip()}
                for r in violations
            ]
        response_obj = {"compliant": compliant, "findings": findings}
        examples.append({
            "image": str(img_path),
            "prompt": f"{PROMPT_PREFIX}\n\n{block}",
            "response": json.dumps(response_obj, ensure_ascii=False),
        })

    save_yolo_cache(cache)
    print(f"可用樣本 {len(examples)}（缺檔 {missing}）")
    random.shuffle(examples)
    n_val = max(20, int(len(examples) * 0.12))
    val = examples[:n_val]
    train = examples[n_val:]

    if TRAIN_VIOLATION_OVERSAMPLE > 1:
        base_viol = [e for e in train if not json.loads(e["response"])["compliant"]]
        train += base_viol * (TRAIN_VIOLATION_OVERSAMPLE - 1)
        random.shuffle(train)
        print(f"[oversample] violation ×{TRAIN_VIOLATION_OVERSAMPLE}：train +{len(base_viol) * (TRAIN_VIOLATION_OVERSAMPLE - 1)} 筆")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    tr_comp = sum(1 for e in train if json.loads(e["response"])["compliant"])
    va_comp = sum(1 for e in val if json.loads(e["response"])["compliant"])
    print(f"train={len(train)}（合規 {tr_comp} / 缺失 {len(train) - tr_comp}）"
          f"  val={len(val)}（合規 {va_comp} / 缺失 {len(val) - va_comp}）")


if __name__ == "__main__":
    main()
