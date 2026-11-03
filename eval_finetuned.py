# -*- coding: utf-8 -*-
"""
用 finetune_data/val.jsonl（57張held-out）評估微調後的 gemma3n-e2b-audit-lora
是否有比基準模型更準。輸出 compliant/findings 的二元判斷（是否有缺失）跟
答案比較，算 accuracy/precision/recall，跟 gemma4:12b 基準（43%）、
gemma4:e4b 隊友測試（52.5%寬鬆準確率）對照。
"""
import io
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "google/gemma-3n-E2B-it"
ADAPTER_DIR = BASE_DIR / "gemma3n-e2b-audit-lora" / "final"
VAL_FILE = BASE_DIR / "finetune_data" / "val.jsonl"

SYSTEM_PROMPT = (
    "你是專業的工地職安稽核員。分析照片後只輸出 JSON："
    '{"compliant": true/false, "findings": [{"item":"項目名稱","result":"缺失描述"}]}'
    "，符合規定時 findings 為空陣列。"
)


def load_val():
    examples = []
    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def predict_compliant(text: str):
    t = text.strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(t[start:end + 1])
    except Exception:
        return None
    return data.get("compliant")


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    val = load_val()
    print(f"驗證集 {len(val)} 筆")

    tp = tn = fp = fn = errors = 0
    rows = []
    for i, ex in enumerate(val, 1):
        gt = json.loads(ex["response"])["compliant"]
        img = Image.open(ex["image"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": ex["prompt"]},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        out_text = processor.decode(gen, skip_special_tokens=True)
        pred = predict_compliant(out_text)

        if pred is None:
            errors += 1
            status = "PARSE_ERROR"
        elif (not pred) and (not gt):
            tp += 1; status = "TP"  # pred=violation, gt=violation
        elif pred and gt:
            tn += 1; status = "TN"  # both compliant
        elif (not pred) and gt:
            fp += 1; status = "FP"  # false alarm
        else:
            fn += 1; status = "FN"  # missed violation

        rows.append({"image": ex["image"], "gt_compliant": gt, "pred_compliant": pred,
                     "status": status, "raw_output": out_text[:500]})
        print(f"[{i}/{len(val)}] gt_compliant={gt} pred_compliant={pred} status={status}")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    summary = {
        "n": len(val), "errors": errors, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4), "recall": round(recall, 4),
    }
    print("\n=== 微調後模型（gemma3n-e2b-audit-lora）評估結果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = BASE_DIR / "eval_finetuned_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"詳細結果已存到 {out_path}")


if __name__ == "__main__":
    main()
