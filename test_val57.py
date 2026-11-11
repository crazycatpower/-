# -*- coding: utf-8 -*-
"""
評測一個 LoRA adapter（或不掛 adapter 的底座）在留出測試集上的整張圖二分類：有缺失／符合規定。

測試集（都沒有進過任何一版的訓練資料）：
  val57   finetune_data/val.jsonl 的 57 張內部照片。歷史版本都用這一份，數字可以直接比。
  ext113  新增資料集/ 的 113 張外部來源照片（勞檢單位手冊），用來看換一批來源還行不行。

流程與 v5~v9 的評測相同：即時跑 YOLO、同一個 system prompt、greedy 解碼、findings 非空就算判定有缺失。
--item 會在 prompt 最後多給「本張照片的稽核項目」（答案表的 D 欄），只有訓練時也給了這行的模型才適用。

用法：
  python test_val57.py gemma3n-e2b-audit-lora-v10/best
  python test_val57.py gemma3n-e2b-audit-lora-v10/best --set ext113
  python test_val57.py base                                # 底座 zero-shot
  python test_val57.py gemma3n-e2b-audit-lora-v10/best --merge   # LoRA 併進 bf16 權重（llm_server.py 的部署方式）
結果寫到 results/<測試集>_<adapter>.json。
"""
import argparse
import json
import math
import time
from pathlib import Path

import audit_vlm_common as C


def load_set(name):
    """回傳 [(圖片路徑, 是否有缺失, 稽核項目提示, 答案文字)]。"""
    if name == "val57":
        answers = C.read_answers(C.ANSWER_XLSX)
        rows = []
        for line in open(C.VAL_JSONL, encoding="utf-8"):
            ex = json.loads(line)
            img = Path(ex["image"].replace("\\", "/")).name
            rows.append((C.IMG_DIR / img, not json.loads(ex["response"])["compliant"],
                         C.item_hint_for(answers[img]), ex["response"]))
        return rows
    answers = C.read_answers(C.EXT_XLSX)
    return [(C.EXT_DIR / img, any(result != C.COMPLIANT for _, result in rows), C.item_hint_for(rows), C.response_for(rows))
            for img, rows in answers.items()]


def wilson(k, n, z=1.96):
    """二項比例的 95% 信賴區間。樣本只有幾十張，單看點估計會誤判差異。"""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", help="adapter 目錄（相對於本檔所在資料夾），或 base")
    ap.add_argument("--set", default="val57", choices=["val57", "ext113"])
    ap.add_argument("--item", action="store_true", help="prompt 多給稽核項目")
    ap.add_argument("--merge", action="store_true", help="LoRA 併進 bf16 權重後再評測")
    ap.add_argument("--out")
    args = ap.parse_args()

    is_base = args.adapter == "base"
    tag = "base" if is_base else "_".join(Path(args.adapter).parts[-2:])
    out_path = Path(args.out) if args.out else C.BASE_DIR / "results" / (
        f"{args.set}_{tag}" + ("_item" if args.item else "") + ("_merged" if args.merge else "") + ".json")
    out_path.parent.mkdir(exist_ok=True)

    targets = load_set(args.set)
    yolo = C.Yolo()
    model, processor = C.load_model(None if is_base else C.BASE_DIR / args.adapter, merge=args.merge)

    count = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "PARSE_ERROR": 0}
    rows = []
    for i, (img_path, gt, item, gt_response) in enumerate(targets, 1):
        t0 = time.perf_counter()
        prompt = C.user_prompt(yolo.block(img_path), item if args.item else None)
        text, p_violation = C.generate_answer(model, processor, C.load_image(img_path), prompt)
        pred = C.predict_violation(text)
        status = "PARSE_ERROR" if pred is None else ("T" if pred == gt else "F") + ("P" if pred else "N")
        count[status] += 1
        rows.append({"image": img_path.name, "gt_violation": gt, "pred_violation": pred, "status": status,
                     "p_violation": None if p_violation is None else round(p_violation, 4),
                     "item": item, "gt_response": gt_response, "output": text[:600]})
        print(f"[{i}/{len(targets)}] {img_path.name} gt={gt} pred={pred} {status} ({time.perf_counter() - t0:.1f}s)", flush=True)

    tp, tn, fp, fn, errors = (count[k] for k in ("TP", "TN", "FP", "FN", "PARSE_ERROR"))
    n, parsed = len(targets), tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    summary = {
        "adapter": args.adapter, "set": args.set, "item_hint": args.item, "merged": args.merge,
        "n": n, "errors": errors, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round((tp + tn) / parsed, 4) if parsed else 0,       # 解析失敗不計（歷史版本的算法）
        "accuracy_strict": round((tp + tn) / n, 4),                      # 解析失敗當成判錯
        "accuracy_strict_ci95": wilson(tp + tn, n),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0,
    }
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
