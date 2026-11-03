# -*- coding: utf-8 -*-
"""
從483張標註資料集隨機抽100張，接上app真實YOLO偵測流程，用v1微調模型
(gemma3n-e2b-audit-lora/final)實際推論，跟答案比對算準確率/precision/recall。
"""
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel

from benchmark import run_yolo
from eval_answer_key import load_answers, IMG_DIR

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = BASE_DIR / "gemma4-e2b-audit-lora-v5" / "final"

_local_snapshot = next(
    (Path.home() / ".cache" / "huggingface" / "hub"
     / "models--google--gemma-3n-E2B-it" / "snapshots").glob("*"),
    None,
)
MODEL_ID = str(_local_snapshot) if _local_snapshot else "google/gemma-3n-E2B-it"
_LOAD_KWARGS = {"local_files_only": True} if _local_snapshot else {}

SYSTEM_PROMPT = (
    "你是專業的工地職安稽核員。分析照片後只輸出 JSON："
    '{"compliant": true/false, "findings": [{"item":"項目名稱","result":"缺失描述"}]}'
    "，符合規定時 findings 為空陣列。"
)


def predict_violation(text):
    t = text.strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None, None
    try:
        data = json.loads(t[s:e + 1])
    except Exception:
        return None, None
    findings = data.get("findings", [])
    return bool(findings), data


def main():
    labels = load_answers()
    all_images = [img for img in labels if (IMG_DIR / img).exists()]
    random.seed(7)
    random.shuffle(all_images)
    targets = all_images[:100]
    print(f"抽樣 {len(targets)} 張")

    processor = AutoProcessor.from_pretrained(MODEL_ID, **_LOAD_KWARGS)
    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16, **_LOAD_KWARGS,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    tp = tn = fp = fn = errors = 0
    rows = []
    for i, name in enumerate(targets, 1):
        img_path = IMG_DIR / name
        gt = labels[name]["violation"]
        t0 = time.perf_counter()
        detections, detection_block = run_yolo(img_path)
        img = Image.open(img_path).convert("RGB")
        prompt_text = f"請分析這張施工現場照片的職安合規性。\n\n{detection_block}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        out_text = processor.decode(gen, skip_special_tokens=True)
        pred, parsed = predict_violation(out_text)
        dt = round(time.perf_counter() - t0, 1)

        if pred is None:
            errors += 1; status = "PARSE_ERROR"
        elif pred and gt:
            tp += 1; status = "TP"
        elif (not pred) and (not gt):
            tn += 1; status = "TN"
        elif pred and (not gt):
            fp += 1; status = "FP"
        else:
            fn += 1; status = "FN"

        rows.append({
            "image": name, "gt_violation": gt, "pred_violation": pred, "status": status,
            "n_detections": len(detections), "raw_answer": labels[name]["raw"],
            "output": out_text[:600],
        })
        print(f"[{i}/{len(targets)}] {name} gt={gt} pred={pred} status={status} ({dt}s)")

    total = tp + tn + fp + fn
    summary = {
        "n": len(targets), "errors": errors, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0,
    }
    print("\n=== v5 模型 100 張真實流程測試結果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    with open(BASE_DIR / "test_100_v5_result.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print("存到 test_100_v5_result.json")


if __name__ == "__main__":
    main()
