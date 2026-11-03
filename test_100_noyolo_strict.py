# -*- coding: utf-8 -*-
"""
同一組100張抽樣圖，這次不接YOLO(跟訓練時的輸入格式完全一致，純圖片+提示詞)，
並加上「內容嚴格比對」：不是只看有無違規判斷對不對，還要看模型講的缺失內容
跟答案是否有實質關聯(字元重疊)，不能只是"剛好都說有問題"就算對。
用v1模型(gemma3n-e2b-audit-lora/final)，方便跟先前的YOLO版本做對比。
"""
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel

from eval_answer_key import load_answers, IMG_DIR

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = BASE_DIR / "gemma3n-e2b-audit-lora" / "final"

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
PROMPT = "請分析這張施工現場照片的職安合規性，判斷是否符合規定，若有缺失請具體列出項目與問題。"

CONTENT_MATCH_THRESHOLD = 0.15  # 字元Jaccard相似度門檻，低於此視為內容不相關


def predict(text):
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


def char_jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def content_matches(gt_texts, findings):
    gt_joined = "".join(gt_texts)
    pred_joined = "".join(
        (f.get("item", "") + f.get("result", "")) for f in findings if isinstance(f, dict)
    )
    return char_jaccard(gt_joined, pred_joined) >= CONTENT_MATCH_THRESHOLD


def main():
    labels = load_answers()
    all_images = [img for img in labels if (IMG_DIR / img).exists()]
    random.seed(7)  # 跟先前YOLO版本用同一個seed，抽到同樣100張，方便對比
    random.shuffle(all_images)
    targets = all_images[:100]
    print(f"抽樣 {len(targets)} 張（跟先前YOLO測試同一批）")

    processor = AutoProcessor.from_pretrained(MODEL_ID, **_LOAD_KWARGS)
    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16, **_LOAD_KWARGS,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    tp = tn = fp = fn = errors = 0
    strict_tp = strict_fp = 0  # 嚴格版：TP裡內容有沒有對上答案
    rows = []
    for i, name in enumerate(targets, 1):
        img_path = IMG_DIR / name
        gt = labels[name]["violation"]
        gt_texts = labels[name]["raw"]
        t0 = time.perf_counter()
        img = Image.open(img_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        out_text = processor.decode(gen, skip_special_tokens=True)
        pred, parsed = predict(out_text)
        dt = round(time.perf_counter() - t0, 1)

        content_ok = None
        if pred is None:
            errors += 1; status = "PARSE_ERROR"
        elif pred and gt:
            tp += 1; status = "TP"
            content_ok = content_matches(gt_texts, parsed.get("findings", []))
            if content_ok:
                strict_tp += 1
            else:
                strict_fp += 1  # 判斷方向對，但內容答非所問，嚴格版不算數
        elif (not pred) and (not gt):
            tn += 1; status = "TN"
        elif pred and (not gt):
            fp += 1; status = "FP"
        else:
            fn += 1; status = "FN"

        rows.append({
            "image": name, "gt_violation": gt, "pred_violation": pred, "status": status,
            "content_ok": content_ok, "raw_answer": gt_texts, "output": out_text[:600],
        })
        print(f"[{i}/{len(targets)}] {name} gt={gt} pred={pred} status={status} "
              f"content_ok={content_ok} ({dt}s)")

    total = tp + tn + fp + fn
    loose_summary = {
        "n": len(targets), "errors": errors, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0,
    }
    # 嚴格版：把「方向對但內容答非所問」的TP改記成FP
    strict_fp_total = fp + strict_fp
    strict_summary = {
        "n": len(targets), "errors": errors, "tp": strict_tp, "tn": tn,
        "fp": strict_fp_total, "fn": fn,
        "accuracy": round((strict_tp + tn) / total, 4) if total else 0,
        "precision": round(strict_tp / (strict_tp + strict_fp_total), 4) if (strict_tp + strict_fp_total) else 0,
        "recall": round(strict_tp / (strict_tp + fn), 4) if (strict_tp + fn) else 0,
    }

    print("\n=== v1 模型 100 張（無YOLO）寬鬆版 ===")
    print(json.dumps(loose_summary, ensure_ascii=False, indent=2))
    print("\n=== v1 模型 100 張（無YOLO）嚴格版（內容需與答案相關）===")
    print(json.dumps(strict_summary, ensure_ascii=False, indent=2))

    with open(BASE_DIR / "test_100_noyolo_result.json", "w", encoding="utf-8") as f:
        json.dump({"loose_summary": loose_summary, "strict_summary": strict_summary, "rows": rows},
                   f, ensure_ascii=False, indent=2)
    print("存到 test_100_noyolo_result.json")


if __name__ == "__main__":
    main()
