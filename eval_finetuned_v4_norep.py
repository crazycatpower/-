# -*- coding: utf-8 -*-
"""
評估 v2 微調模型（gemma4-e2b-audit-lora-v4，描述品質提升+強制繁體中文版）
在同一組57張held-out驗證集上的表現，跟 v1（80.7%準確率）、基準比較。
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent
_local_snapshot = next(
    (Path.home() / ".cache" / "huggingface" / "hub"
     / "models--google--gemma-3n-E2B-it" / "snapshots").glob("*"),
    None,
)
MODEL_ID = str(_local_snapshot) if _local_snapshot else "google/gemma-3n-E2B-it"
_LOAD_KWARGS = {"local_files_only": True} if _local_snapshot else {}
ADAPTER_DIR = BASE_DIR / "gemma4-e2b-audit-lora-v4" / "final"
VAL_FILE = BASE_DIR / "finetune_data" / "val.jsonl"

SYSTEM_PROMPT = (
    "你是專業的工地職安稽核員。分析照片後只輸出 JSON："
    '{"compliant": true/false, "summary":"一句話總結", "findings":["項目：缺失描述", ...]}'
    "，符合規定時 findings 為空陣列。"
    "所有文字內容必須全部使用繁體中文，禁止出現任何簡體字。"
)


def load_val():
    examples = []
    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(t[start:end + 1])
    except Exception:
        return None


def has_simplified(text: str):
    # 常見簡體/繁體差異字快篩（非完整，但夠用來抽查）
    simp_markers = "国说这里发现问题级现应该关达为区门实应经过还没时对进认与从会应经检测统计"
    return any(ch in text for ch in simp_markers)


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, **_LOAD_KWARGS)
    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16, **_LOAD_KWARGS,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    val = load_val()
    print(f"驗證集 {len(val)} 筆")

    tp = tn = fp = fn = errors = 0
    simplified_count = 0
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
        parsed = parse_json(out_text)
        pred = parsed.get("compliant") if parsed else None

        if has_simplified(out_text):
            simplified_count += 1

        if pred is None:
            errors += 1
            status = "PARSE_ERROR"
        else:
            # violation(=not compliant) 當正類，跟之前v1/基準的算法一致
            pred_violation, gt_violation = not pred, not gt
            if pred_violation and gt_violation:
                tp += 1; status = "TP"
            elif (not pred_violation) and (not gt_violation):
                tn += 1; status = "TN"
            elif pred_violation and (not gt_violation):
                fp += 1; status = "FP"
            else:
                fn += 1; status = "FN"

        rows.append({"image": ex["image"], "gt_compliant": gt, "pred_compliant": pred,
                     "status": status, "raw_output": out_text[:800]})
        print(f"[{i}/{len(val)}] gt_compliant={gt} pred_compliant={pred} status={status}")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    summary = {
        "n": len(val), "errors": errors, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4), "recall": round(recall, 4),
        "simplified_suspect_count": simplified_count,
    }
    print("\n=== v2 微調後模型（gemma4-e2b-audit-lora-v4）評估結果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = BASE_DIR / "eval_finetuned_v4_norep_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"詳細結果已存到 {out_path}")


if __name__ == "__main__":
    main()
