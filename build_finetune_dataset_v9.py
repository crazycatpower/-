# -*- coding: utf-8 -*-
"""
v9：在 8 小時內能訓練完的「最好一版」資料集。val 仍沿用 finetune_data/val.jsonl（57 筆，與訓練圖零重疊）。

和 v8 的差別：
  1. real 的 YOLO 清單全部用目前的 best.pt 重跑。v5~v8 的 real 列沿用 finetune_data/yolo_cache.json，
     那是舊版 best.pt 的輸出（抽查 8 張只有 1 張和現在一致），訓練輸入和推論輸入不一致。
  2. 每張原圖只取 1 份增強（__aug0），不是 2 份：同一場景重複太多次只是在背答案，
     把步數留給多樣性。違規樣本照 v6 複製 TRAIN_VIOLATION_OVERSAMPLE 份（原圖與增強圖都複製）。
  3. web 照片（125 張人工檢視標註）= 原圖 + __aug0，違規同樣複製。
  4. 訓練 1 個 epoch（finetune_gemma3n_v9.py）：每張 real 照片在一個 epoch 內被看到 2~4 次，
     和 v6「3 epoch」的曝光量相當，但總步數約 180 步，RTX 3060 約 4~5 小時。

張晴茵的 test_rag.jsonl：內部 100 張與 答案.xlsx 判定 99/99 一致（同一份答案轉出），沒有新資訊；
外部 57 張（kcg_hazard）本機沒有圖檔，取得後可加進 web 來源。

輸出：finetune_data_v9/{train,val}.jsonl、stats.json。
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import build_finetune_dataset_v8 as v8

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "finetune_data"
OUT_DIR = BASE_DIR / "finetune_data_v9"
AUG_DIR = BASE_DIR / "uploads" / "augmented"
IMG_DIR = BASE_DIR / "uploads" / "UploadImg"
YOLO_CACHE = OUT_DIR / "yolo_cache.json"
V8_CACHE = BASE_DIR / "finetune_data_v8" / "yolo_cache.json"  # 增強圖與 web 圖已用目前 best.pt 跑過
TRAIN_VIOLATION_OVERSAMPLE = 2


def main():
    rng = random.Random(42)
    OUT_DIR.mkdir(exist_ok=True)
    cache = {}
    for p in (V8_CACHE, YOLO_CACHE):
        if p.exists():
            cache.update(json.loads(p.read_text(encoding="utf-8")))
    yolo = v8.Yolo()

    def block_for(path: Path):
        key = str(path)
        if key not in cache:
            cache[key] = yolo.block(path)
            if len(cache) % 100 == 0:
                YOLO_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return cache[key]

    def example(path: Path, resp: dict, source: str):
        return {"image": str(path), "prompt": f"{v8.PROMPT_PREFIX}\n\n{block_for(path)}",
                "response": json.dumps(resp, ensure_ascii=False), "source": source}

    def add(rows, path, resp, source):
        reps = 1 if resp["compliant"] else TRAIN_VIOLATION_OVERSAMPLE
        rows += [example(path, resp, source)] * reps

    val = v8.read_jsonl(SRC_DIR / "val.jsonl")
    val_imgs = {Path(e["image"]).name for e in val}

    # real：每張 train 原圖一次（原 train.jsonl 已含過採樣，這裡先去重再自己過採樣）
    uniq = {}
    for e in v8.read_jsonl(SRC_DIR / "train.jsonl"):
        uniq.setdefault(e["image"], json.loads(e["response"]))
    assert not ({Path(p).name for p in uniq} & val_imgs), "train/val 圖片重疊"

    train = []
    for i, (img, resp) in enumerate(sorted(uniq.items())):
        src = IMG_DIR / Path(img).name
        add(train, src, resp, "real")
        aug = AUG_DIR / f"{src.stem}__aug0.jpg"
        if not aug.exists():
            v8.augment(src, aug, rng)
        add(train, aug, resp, "aug")
        if (i + 1) % 50 == 0:
            print(f"[real+aug] {i + 1}/{len(uniq)}", flush=True)

    for ip, resp in v8.load_web():
        add(train, ip, resp, "web")
        aug = AUG_DIR / f"{ip.stem}__aug0.jpg"
        if not aug.exists():
            v8.augment(ip, aug, rng)
        add(train, aug, resp, "web")

    YOLO_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    rng.shuffle(train)
    with open(OUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for e in train:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    shutil.copyfile(SRC_DIR / "val.jsonl", OUT_DIR / "val.jsonl")

    def summarize(rows):
        resp = [json.loads(e["response"]) for e in rows]
        return {"rows": len(rows), "unique_images": len({e["image"] for e in rows}),
                "compliant": sum(r["compliant"] for r in resp), "violation": sum(not r["compliant"] for r in resp),
                "top_findings": Counter(f"{f['item']}｜{f['result']}" for r in resp for f in r["findings"]).most_common(15)}

    stats = {"train_total": summarize(train), "val": summarize(val),
             **{f"train_{s}": summarize([e for e in train if e["source"] == s]) for s in ("real", "aug", "web")}}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in stats.items():
        print(f"{k:12s} rows={v['rows']:5d} 圖={v['unique_images']:5d} 合規={v['compliant']:5d} 缺失={v['violation']:5d}")


if __name__ == "__main__":
    main()
