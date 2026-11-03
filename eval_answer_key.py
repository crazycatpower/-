# -*- coding: utf-8 -*-
"""
用答案.xlsx 評估目前 LLM（gemma4:12b）在真實工地照片上的稽核準確率。
只做二元判斷：這張照片答案裡是否「全部符合規定」vs「至少一項違規」，
和 LLM 輸出的 categories 是否至少有一項非空（含 × 或 △）比對。

用法:
  python eval_answer_key.py --sample 60          # 隨機抽樣60張
  python eval_answer_key.py --all                # 跑全部483張（很慢）
  python eval_answer_key.py --sample 60 --seed 1
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import openpyxl
import requests
from dotenv import load_dotenv
load_dotenv()

from benchmark import (
    BASE_SYSTEM_PROMPT,
    run_yolo,
    image_to_data_url,
)

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "uploads" / "UploadImg"
ANSWER_XLSX = BASE_DIR / "答案.xlsx"


def load_answers():
    wb = openpyxl.load_workbook(ANSWER_XLSX)
    ws = wb.worksheets[0]
    per_image = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        base = os.path.basename(str(row[0]).replace("\\", "/"))
        result_text = row[4]
        per_image.setdefault(base, []).append(result_text)

    labels = {}
    for img, results in per_image.items():
        is_violation = any(r != "符合規定" for r in results)
        labels[img] = {"violation": is_violation, "raw": results}
    return labels


def predict_violation(llm_json_text: str):
    """從 LLM 回覆解析出是否判定違規；解析失敗回傳 None。"""
    text = llm_json_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None, None
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return None, None

    categories = data.get("categories", {})
    has_violation = False
    for items in categories.values():
        if not isinstance(items, list):
            continue
        for it in items:
            if isinstance(it, dict) and it.get("result") in ("×", "△"):
                has_violation = True
    return has_violation, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "gemma4:12b"))
    args = ap.parse_args()

    labels = load_answers()
    all_images = [img for img in labels if (IMG_DIR / img).exists()]
    print(f"答案中共 {len(labels)} 張圖，實際存在檔案 {len(all_images)} 張")

    random.seed(args.seed)
    random.shuffle(all_images)
    targets = all_images if args.all else all_images[: args.sample]
    print(f"本次評估 {len(targets)} 張，模型={args.model}")

    native_url = os.getenv("OLLAMA_NATIVE_URL", "http://127.0.0.1:11434/api/chat")

    rows = []
    tp = tn = fp = fn = errors = 0

    for i, img in enumerate(targets, 1):
        img_path = IMG_DIR / img
        gt = labels[img]["violation"]
        t0 = time.perf_counter()
        try:
            _, detection_block = run_yolo(img_path)
            data_url = image_to_data_url(img_path)
            b64_only = data_url.split(",", 1)[1]
            messages = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"請分析這張施工現場照片的職安合規性。\n\n{detection_block}",
                    "images": [b64_only],
                },
            ]
            resp = requests.post(
                native_url,
                json={
                    "model": args.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 2048},
                },
                timeout=180,
            )
            resp.raise_for_status()
            out_text = resp.json().get("message", {}).get("content", "") or ""
            pred, parsed = predict_violation(out_text)
        except Exception as e:
            pred, parsed, out_text = None, None, f"ERROR: {e}"

        dt = round(time.perf_counter() - t0, 1)

        if pred is None:
            errors += 1
            status = "PARSE_ERROR"
        elif pred and gt:
            tp += 1; status = "TP"
        elif (not pred) and (not gt):
            tn += 1; status = "TN"
        elif pred and (not gt):
            fp += 1; status = "FP"
        else:
            fn += 1; status = "FN"

        rows.append({
            "image": img, "ground_truth_violation": gt, "pred_violation": pred,
            "status": status, "seconds": dt, "raw_answer": labels[img]["raw"],
            "llm_output": out_text,
        })
        print(f"[{i}/{len(targets)}] {img} gt={gt} pred={pred} status={status} ({dt}s)")

    total_scored = tp + tn + fp + fn
    accuracy = (tp + tn) / total_scored if total_scored else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    summary = {
        "model": args.model, "n": len(targets), "errors": errors,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4), "recall": round(recall, 4),
    }
    print("\n=== 結果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = BASE_DIR / f"eval_result_{args.model.replace(':', '_')}_{len(targets)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果已存到 {out_path}")


if __name__ == "__main__":
    main()
