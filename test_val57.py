# -*- coding: utf-8 -*-
"""
在 finetune_data/val.jsonl 的 57 張上評測一個 LoRA adapter。

為什麼不用 test_100：test_100 從全部標註隨機抽 100 張，事後檢查有 91 張是 v6 訓練過的，分數偏高。
val 的 57 張在 v5~v9 的訓練資料裡都沒有出現（v8/v9 的增強與網路照片也不含這些圖），是唯一乾淨的測試集。

流程與 test_100_real_flow_v6.py 相同（即時跑 YOLO、同一個 system prompt、greedy 解碼、整張圖二分類），
差別只有：圖片先依 EXIF 轉正（YOLO 的 cv2.imread 會轉正，VLM 這邊也要一致）。

用法：python test_val57.py gemma4-e2b-audit-lora-v9/final  [輸出檔名]
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel

from benchmark import run_yolo
from test_100_real_flow_v6 import MODEL_ID, _LOAD_KWARGS, SYSTEM_PROMPT, predict_violation

BASE_DIR = Path(__file__).resolve().parent
VAL = BASE_DIR / "finetune_data" / "val.jsonl"
IMG_DIR = BASE_DIR / "uploads" / "UploadImg"


def main():
    adapter = BASE_DIR / sys.argv[1]
    out_path = BASE_DIR / (sys.argv[2] if len(sys.argv) > 2 else f"val57_{adapter.parent.name}_{adapter.name}.json")
    targets = [json.loads(l) for l in open(VAL, encoding="utf-8")]

    processor = AutoProcessor.from_pretrained(MODEL_ID, **_LOAD_KWARGS)
    base = Gemma3nForConditionalGeneration.from_pretrained(MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16, **_LOAD_KWARGS)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    tp = tn = fp = fn = errors = 0
    rows = []
    for i, ex in enumerate(targets, 1):
        img_path = IMG_DIR / Path(ex["image"]).name
        gt = not json.loads(ex["response"])["compliant"]
        t0 = time.perf_counter()
        detections, block = run_yolo(img_path)
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": f"請分析這張施工現場照片的職安合規性。\n\n{block}"}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        out_text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred, _ = predict_violation(out_text)
        if pred is None:
            errors += 1; status = "PARSE_ERROR"
        elif pred and gt:
            tp += 1; status = "TP"
        elif not pred and not gt:
            tn += 1; status = "TN"
        elif pred:
            fp += 1; status = "FP"
        else:
            fn += 1; status = "FN"
        rows.append({"image": img_path.name, "gt_violation": gt, "pred_violation": pred, "status": status,
                     "gt_response": ex["response"], "output": out_text[:600]})
        print(f"[{i}/{len(targets)}] {img_path.name} gt={gt} pred={pred} {status} ({time.perf_counter() - t0:.1f}s)", flush=True)

    total = tp + tn + fp + fn
    # 解析失敗另外算一版「當成判錯」的嚴格指標，避免靠輸出壞掉來躲掉難題
    strict_acc = (tp + tn) / len(targets)
    summary = {"adapter": sys.argv[1], "n": len(targets), "errors": errors, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "accuracy": round((tp + tn) / total, 4) if total else 0,
               "accuracy_strict": round(strict_acc, 4),
               "precision": round(tp / (tp + fp), 4) if tp + fp else 0,
               "recall": round(tp / (tp + fn), 4) if tp + fn else 0}
    summary["f1"] = round(2 * summary["precision"] * summary["recall"] / (summary["precision"] + summary["recall"]), 4) \
        if summary["precision"] + summary["recall"] else 0
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
