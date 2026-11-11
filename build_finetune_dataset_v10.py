# -*- coding: utf-8 -*-
"""
v10 訓練資料。資料來源與 v9 相同（內部照片 + 人工標註的網路照片），差別在怎麼切、怎麼排：

  1. 測試集完全不碰：finetune_data/val.jsonl 的 57 張（歷史版本共用），以及 新增資料集/ 的 113 張外部照片。
     外部照片留作第二個測試集而不是拿來訓練——它同時是晴茵那邊的測試集，訓練進去兩邊就不能比了。
  2. 從訓練用的內部照片再切 DEV_SIZE 張當 dev，訓練中用它挑 checkpoint。
     以前是拿 val57 挑 checkpoint 又拿 val57 報分數，分數會偏高。
  3. 每張圖有三個版本（原圖、__aug0、__aug1），輪到第幾個 epoch 用哪一版由訓練腳本決定，
     同一張圖每個 epoch 看到的樣子都不同；缺失樣本的過採樣也在訓練腳本做。
  4. YOLO 偵測清單全部用目前的 best.pt 對「實際送進模型的那張圖」重跑，格式與評測逐字相同。
     （舊版用 cv2.imread 取圖片尺寸，檔名含中文的圖讀不到，座標沒有正規化。）
  5. 每筆多存一個 item 欄位（答案表的稽核項目），訓練腳本開 --item 時會接在 prompt 後面。

輸出：finetune_data_v10/{train,dev}.jsonl、stats.json。圖片路徑是相對於本資料夾的相對路徑。
"""
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import audit_vlm_common as C

OUT_DIR = C.BASE_DIR / "finetune_data_v10"
AUG_DIR = C.BASE_DIR / "uploads" / "augmented"
WEB_LABELS = C.BASE_DIR / "external_data" / "web_labels.tsv"
WEB_DIRS = {"wc": "web_commons", "ov": "web_openverse", "ox": "web_openverse2"}
DEV_SIZE = 48
AUG_VERSIONS = 2          # 每張圖除了原圖之外的增強版本數
AUG_MAX_CROP = 0.06       # 每邊最多裁掉的比例
AUG_SAVE_MAX_SIDE = 1600


def augment(src, dst, rng):
    """只做不會改變稽核結論的增強：輕微裁切、亮度／對比／飽和度、輕微模糊、JPEG 重壓。
    不做水平翻轉——告示牌和交通標誌的文字翻過去就不對了。"""
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = img.size
    left, top = rng.uniform(0, AUG_MAX_CROP) * w, rng.uniform(0, AUG_MAX_CROP) * h
    right, bottom = w - rng.uniform(0, AUG_MAX_CROP) * w, h - rng.uniform(0, AUG_MAX_CROP) * h
    img = img.crop((int(left), int(top), int(right), int(bottom)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    img.thumbnail((AUG_SAVE_MAX_SIDE, AUG_SAVE_MAX_SIDE))
    img.save(dst, "JPEG", quality=rng.randint(70, 92))


def load_web():
    """external_data/web_labels.tsv → [(圖片路徑, [(項目, 結果), ...])]；skip 的不收。"""
    out = []
    for line in WEB_LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        img_id, label, findings = (line.split("\t") + ["", ""])[:3]
        if label == "skip":
            continue
        rows = [tuple(f.split("｜")) for f in findings.split(";;")] if label == "violation" else [("", C.COMPLIANT)]
        out.append((C.BASE_DIR / "external_data" / WEB_DIRS[img_id[:2]] / f"{img_id}.jpg", rows))
    return out


def main():
    rng = random.Random(42)
    OUT_DIR.mkdir(exist_ok=True)
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    yolo = C.Yolo()

    answers = C.read_answers(C.ANSWER_XLSX)
    val = {Path(json.loads(l)["image"].replace("\\", "/")).name for l in open(C.VAL_JSONL, encoding="utf-8")}
    real = sorted((C.IMG_DIR / name, rows) for name, rows in answers.items()
                  if name not in val and (C.IMG_DIR / name).exists())

    # dev：依有無缺失分層抽樣
    by_label = {True: [], False: []}
    for path, rows in real:
        by_label[any(r != C.COMPLIANT for _, r in rows)].append(path.name)
    dev = set()
    for label, names in by_label.items():
        rng.shuffle(names)
        dev |= set(names[: round(DEV_SIZE * len(names) / len(real))])

    def record(path, rows, source, group, version):
        """group：同一張原圖的各版本共用；version：0 原圖、1 以上是增強版本。"""
        return {"image": path.relative_to(C.BASE_DIR).as_posix(), "group": group, "version": version,
                "source": source, "prompt": C.user_prompt(yolo.block(path)), "item": C.item_hint_for(rows),
                "response": C.response_for(rows)}

    train_rows, dev_rows = [], []
    pool = [(p, r, "real") for p, r in real] + [(p, r, "web") for p, r in load_web()]
    assert len({p.stem for p, _, _ in pool}) == len(pool), "有兩張圖主檔名相同，增強圖檔名會互相覆蓋"
    for i, (path, rows, source) in enumerate(pool, 1):
        if source == "real" and path.name in dev:
            dev_rows.append(record(path, rows, source, path.stem, 0))
            continue
        train_rows.append(record(path, rows, source, path.stem, 0))
        for k in range(AUG_VERSIONS):
            aug = AUG_DIR / f"{path.stem}__aug{k}.jpg"
            if not aug.exists():
                augment(path, aug, rng)
            train_rows.append(record(aug, rows, source, path.stem, k + 1))
        if i % 50 == 0:
            print(f"[build] {i}/{len(pool)}", flush=True)

    for name, data in (("train", train_rows), ("dev", dev_rows)):
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def summarize(data):
        originals = [e for e in data if e["version"] == 0]
        resp = [json.loads(e["response"]) for e in originals]
        return {"rows": len(data), "images": len(originals),
                "compliant": sum(r["compliant"] for r in resp), "violation": sum(not r["compliant"] for r in resp),
                "top_findings": Counter(f"{f['item']}｜{f['result']}" for r in resp for f in r["findings"]).most_common(15)}

    stats = {"train": summarize(train_rows), "dev": summarize(dev_rows),
             **{f"train_{s}": summarize([e for e in train_rows if e["source"] == s]) for s in ("real", "web")}}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in stats.items():
        print(f"{k:11s} rows={v['rows']:5d} 圖={v['images']:4d} 合規={v['compliant']:4d} 缺失={v['violation']:4d}")


if __name__ == "__main__":
    main()
