# -*- coding: utf-8 -*-
"""
用主程式原本的流程（safety_audit_system.analyze_image：YOLO、原圖＋標註圖、54 條清單與法規 prompt、後處理）
讓 Ollama 上的大模型跑 val57 / ext113，和 v10 比較。

判定：稽核表任何一條被打 × 或 △ → 有缺失（與 v10「findings 非空」對應）；另外記錄只算 × 的判定。
結果寫 results/<set>_ollama-<model>.json，欄位與 test_val57.py 相同（compare_fall.py 可直接用），
每張存一次，中斷後重跑會跳過已完成的照片。

用法：python benchmark_bigmodel.py gemma4:12b [--set val57|ext113]
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

from openai import OpenAI

import audit_vlm_common as C
import safety_audit_system as S
from test_val57 import load_set, wilson

OLLAMA = "http://127.0.0.1:11434/v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--set", default="val57", choices=["val57", "ext113"])
    args = ap.parse_args()

    # 主程式每次都從 .env 讀模型設定（override=True）；這裡固定成要比較的模型，其餘流程不動
    S.get_llm_client_and_config = lambda: (OpenAI(api_key="ollama", base_url=OLLAMA), args.model, True)
    S.os.environ["LLM_BASE_URL"] = OLLAMA

    out_path = C.BASE_DIR / "results" / f"{args.set}_ollama-{args.model.replace(':', '-')}.json"
    done = {r["image"]: r for r in json.load(open(out_path, encoding="utf-8"))["rows"]} if out_path.exists() else {}
    work = Path(tempfile.mkdtemp(prefix="bigmodel_"))
    targets = load_set(args.set)

    rows = []
    for i, (img_path, gt, item, gt_response) in enumerate(targets, 1):
        if img_path.name in done:
            rows.append(done[img_path.name])
            continue
        t0 = time.perf_counter()
        res = S.analyze_image(img_path, yolo_model_path="best.pt", llm_model=args.model, output_dir=work)
        scene = res["scene_analysis"]
        cats = scene.get("categories") if isinstance(scene.get("categories"), dict) else {}
        flagged = [(it.get("item"), it.get("result")) for items in cats.values() if isinstance(items, list)
                   for it in items if isinstance(it, dict) and it.get("result") in ("×", "△")]
        ok = bool(cats)
        pred = bool(flagged) if ok else None
        pred_x = any(r == "×" for _, r in flagged) if ok else None
        status = "PARSE_ERROR" if pred is None else ("T" if pred == gt else "F") + ("P" if pred else "N")
        rows.append({"image": img_path.name, "gt_violation": gt, "pred_violation": pred, "pred_violation_x_only": pred_x,
                     "status": status, "p_violation": None, "item": item, "gt_response": gt_response,
                     "flagged": flagged, "summary": str(scene.get("summary") or "")[:300],
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[{i}/{len(targets)}] {img_path.name[:32]} gt={gt} pred={pred} {status} 標記={flagged[:4]} "
              f"({rows[-1]['seconds']}s)", flush=True)
        save(out_path, args, rows)
    save(out_path, args, rows)


def save(out_path, args, rows):
    count = {k: sum(r["status"] == k for r in rows) for k in ("TP", "TN", "FP", "FN", "PARSE_ERROR")}
    n = len(rows)
    x_only = [r for r in rows if r.get("pred_violation_x_only") is not None]
    summary = {"model": args.model, "set": args.set, "n": n, **{k.lower(): v for k, v in count.items()},
               "accuracy_strict": round((count["TP"] + count["TN"]) / n, 4) if n else 0,
               "accuracy_strict_ci95": wilson(count["TP"] + count["TN"], n) if n else None,
               "accuracy_x_only": round(sum(r["pred_violation_x_only"] == r["gt_violation"] for r in x_only) / n, 4) if n else 0}
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
