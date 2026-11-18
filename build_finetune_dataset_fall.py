# -*- coding: utf-8 -*-
"""
墜落防止（稽核表第 1 大類，1.01~1.08）專用訓練資料，給 finetune_gemma3n_fall.py 用。

來源：
  1. 答案.xlsx 裡大類編號 = 1 的內部照片。沿用 finetune_data_v10 的切法：v10 train 的留在 train，
     v10 dev 的留在 dev，val57 仍然完全不碰。
  2. external_data/web_labels.tsv 裡和墜落有關的網路照片（已在 v10 train 裡）。
  3. external_data/fall_labels.tsv：從公開手冊（臺中、彰化查核缺失手冊、職安署、職災預防中心、高雄屋頂作業指引）
     抽出、依圖說標註的照片。外部測試集 ext113 用過的 4 份手冊（新北墜落預防、高雄危害辨識／管溝／工地管理手冊）
     不在其中。約四分之一（依合規／缺失分層）留作 dev。

ext113 的墜落照片仍是測試集，不進訓練也不拿來挑 checkpoint。

輸出 finetune_data_fall/{train,dev}.jsonl、stats.json，欄位與 finetune_data_v10 相同。
"""
import json
import random
from collections import Counter
from pathlib import Path

import openpyxl

import audit_vlm_common as C
from build_finetune_dataset_v10 import AUG_DIR, AUG_VERSIONS, augment

OUT_DIR = C.BASE_DIR / "finetune_data_fall"
V10_DIR = C.BASE_DIR / "finetune_data_v10"
FALL_LABELS = C.BASE_DIR / "external_data" / "fall_labels.tsv"
WEB_LABELS = C.BASE_DIR / "external_data" / "web_labels.tsv"
FALL_ITEM = "於高差2公尺以上之工作場所邊緣及開口部分"
FALL_KEYWORDS = ("高差", "於2公尺以上", "護欄高度", "易踏穿", "鋼構建築")
# 網路照片裡合規、而且畫面重點是墜落防護的（web_labels.tsv 的合規列沒有記項目，只能逐張挑）
WEB_FALL_COMPLIANT = {"wc0039", "wc0406", "wc0559", "wc0562", "wc0821", "ov0120", "ov0121",
                      "ox0002", "ox0004", "ox0007", "ox0027", "ox0028"}
DEV_FRACTION = 0.25


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def internal_fall_names():
    """答案.xlsx 大類編號（B 欄）= 1 的圖檔名。"""
    names = set()
    for row in openpyxl.load_workbook(C.ANSWER_XLSX).worksheets[0].iter_rows(values_only=True):
        if row[0] and row[1] == 1:
            names.add(str(row[0]).replace("\\", "/").split("/")[-1])
    return names


def web_fall_ids():
    ids = set(WEB_FALL_COMPLIANT)
    for line in WEB_LABELS.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == "violation" and any(k in parts[2] for k in FALL_KEYWORDS):
            ids.add(parts[0])
    return ids


def main():
    rng = random.Random(7)
    OUT_DIR.mkdir(exist_ok=True)
    fall_stems = {Path(n).stem for n in internal_fall_names()} | web_fall_ids()

    train, dev = [], []
    for row in read_jsonl(V10_DIR / "train.jsonl"):
        if row["group"] in fall_stems:
            if row["group"] in WEB_FALL_COMPLIANT:
                row = {**row, "item": FALL_ITEM}  # 合規列原本沒有項目
            train.append(row)
    dev += [r for r in read_jsonl(V10_DIR / "dev.jsonl") if r["group"] in fall_stems]

    # 手冊照片：依合規／缺失分層，約四分之一留作 dev
    labeled = []
    for line in FALL_LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rel, label, _code, item, result = line.split("\t")[:5]
        labeled.append((C.BASE_DIR / "external_data" / "fall_src" / f"{rel}.jpg", label, item, result))
    dev_paths = set()
    for label in ("compliant", "violation"):
        group = [x for x in labeled if x[1] == label]
        rng.shuffle(group)
        dev_paths |= {x[0] for x in group[: round(len(group) * DEV_FRACTION)]}

    yolo = C.Yolo()

    def record(path, rows, item, group, version):
        return {"image": path.relative_to(C.BASE_DIR).as_posix(), "group": group, "version": version,
                "source": "manual", "prompt": C.user_prompt(yolo.block(path)), "item": item,
                "response": C.response_for(rows)}

    for path, label, item, result in labeled:
        rows = [(item, C.COMPLIANT if label == "compliant" else result)]
        item_hint = item.rstrip("：:，,").strip()
        group = f"fall_{path.stem}"
        if path in dev_paths:
            dev.append(record(path, rows, item_hint, group, 0))
            continue
        train.append(record(path, rows, item_hint, group, 0))
        for k in range(AUG_VERSIONS):
            aug = AUG_DIR / f"fall_{path.stem}__aug{k}.jpg"
            if not aug.exists():
                augment(path, aug, rng)
            train.append(record(aug, rows, item_hint, group, k + 1))

    for name, data in (("train", train), ("dev", dev)):
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def summarize(data):
        originals = [e for e in data if e["version"] == 0]
        resp = [json.loads(e["response"]) for e in originals]
        return {"rows": len(data), "images": len(originals),
                "compliant": sum(r["compliant"] for r in resp), "violation": sum(not r["compliant"] for r in resp),
                "by_source": Counter(e["source"] for e in originals)}

    stats = {"train": summarize(train), "dev": summarize(dev)}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
