# -*- coding: utf-8 -*-
"""
v8：擴充訓練集。val 完全沿用 finetune_data/val.jsonl（57 筆），指標才能跟 v5~v7 直接比較。

train = 三個來源合併：
  1. real   ：finetune_data/train.jsonl 原樣保留（v6/v7 的 620 筆，含缺失過採樣 ×2）。
  2. aug    ：對 train split 的 426 張原圖各做 AUG_PER_IMAGE 個增強版本，沿用原圖標籤。
              只做不會改變稽核結論的增強：輕微裁切（每邊最多裁 AUG_MAX_CROP）、亮度／對比／
              飽和度擾動、輕微模糊、JPEG 重壓。不做水平翻轉——工程告示牌、交通標誌的文字
              翻轉後內容就不對了。val 的圖完全不參與增強，避免資料外洩。
  3. cs10k  ：ConstructionSite 10k（LouisChen15/ConstructionSite，HF gated，CC BY-NC 4.0，
              影像取自 MOCS）。它的人工違規標註 rule_1~4 中，2/3/4 條是「違規操作／設施缺失」，
              直接對到 答案.xlsx 的稽核項目原文（見 CS10K_RULE_MAP）。Rule 1 是 PPE，裝備交給
              YOLO 判斷，不轉成 finding；只違反 Rule 1 的圖整張跳過，避免「圖上有人沒戴帽卻標合規」
              的矛盾標籤。四條都沒違反的圖當合規樣本，數量另設上限。
  5. web    ：Wikimedia Commons／Openverse 上 CC BY、CC BY-SA、CC0 授權的工地照片（不收 NC、ND），
              逐張人工檢視後依 答案.xlsx 原文標註，標註檔是 external_data/web_labels.tsv
              （skip 的不收）。每張圖跟 real 一樣做 AUG_PER_IMAGE 份增強；缺失樣本跟 v6 一樣
              複製 TRAIN_VIOLATION_OVERSAMPLE 份。來源與授權見 external_data/web_attribution.csv。
  4. ppe    ：（預設關閉，USE_PPE_EXT）VincentGOURBIN/ppe-detection（CC BY 4.0），
              no-helmet / no-vest 轉成「進入工區｜未佩戴安全帽／未穿反光背心」。
              裝備已由 YOLO 負責，主任務是違規操作，所以不加。

不論哪個來源，prompt 裡的 YOLO detection_block 一律對「實際送進模型的那張圖」重跑 best.pt，
格式與 benchmark.run_yolo 逐字相同（不是拿人工標註框假裝成 YOLO 輸出），跟推論時的輸入分布一致。

輸出：finetune_data_v8/{train,val}.jsonl、stats.json；增強圖存在 uploads/augmented/。
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "finetune_data"
OUT_DIR = BASE_DIR / "finetune_data_v8"
AUG_DIR = BASE_DIR / "uploads" / "augmented"
EXT_DIR = BASE_DIR / "external_data" / "ppe_detection"
CS10K_DIR = BASE_DIR / "external_data" / "constructionsite"   # hf download 的 local_dir
CS10K_IMG_DIR = CS10K_DIR / "images"                         # 從 parquet 解出來的 jpg
YOLO_CACHE = OUT_DIR / "yolo_cache.json"
WEB_LABELS = BASE_DIR / "external_data" / "web_labels.tsv"
WEB_DIRS = {"wc": BASE_DIR / "external_data" / "web_commons", "ov": BASE_DIR / "external_data" / "web_openverse",
            "ox": BASE_DIR / "external_data" / "web_openverse2"}
TRAIN_VIOLATION_OVERSAMPLE = 2  # 與 build_finetune_dataset.py v6 相同

PROMPT_PREFIX = "請分析這張施工現場照片的職安合規性。"
EXT_ITEM = "進入工區"
EXT_NO_HELMET = "未佩戴安全帽"
EXT_NO_VEST = "未穿反光背心"

AUG_PER_IMAGE = 2       # 每張 train 原圖產生幾個增強版本
AUG_MAX_CROP = 0.06     # 每邊最多裁掉的比例
AUG_SAVE_MAX_SIDE = 1600  # 增強圖存檔時長邊上限（原圖最大 4032，存小一點省空間與訓練時間）
USE_PPE_EXT = False
EXT_MAX_VIOLATION = 700   # PPE 外部資料最多取幾筆缺失樣本
EXT_MAX_COMPLIANT = 250   # PPE 外部資料最多取幾筆合規樣本

FALL_ITEM = "於高差2公尺以上之工作場所邊緣及開口部分，"
CS10K_RULE_MAP = {
    "rule_2_violation": {"item": FALL_ITEM, "result": "未配掛安全帶之防墜設施"},
    "rule_3_violation": {"item": FALL_ITEM, "result": "未設置符合規定之護欄、護蓋、安全網"},
    "rule_4_violation": {"item": "除乘坐席外，作業時不得搭載勞工；作業半徑內", "result": "未禁止人員進入"},
}
CS10K_MAX_VIOLATION = 1500  # CS10k 全取（違規樣本本來就不多）
CS10K_MAX_COMPLIANT = 300   # 合規樣本壓低比例，避免稀釋真實資料的缺失分布

_REGIONS = (("上左", "上中", "上右"), ("中左", "中央", "中右"), ("下左", "下中", "下右"))


class Yolo:
    """與 benchmark.run_yolo 相同的偵測與 detection_block 格式，但模型只載一次。"""

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(str(BASE_DIR / "best.pt"))

    def block(self, image_path: Path) -> str:
        import cv2
        results = self.model.predict(source=str(image_path), conf=0.25, save=False, verbose=False)
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2] if img is not None else (1, 1)
        lines = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                region = _REGIONS[min(2, int(cy * 3))][min(2, int(cx * 3))]
                lines.append(
                    f"  #{len(lines) + 1} 類別={self.model.names[int(box.cls[0])]} "
                    f"信心={float(box.conf[0]):.2f} 區域={region} "
                    f"norm_xyxy=({round(x1 / w, 4)},{round(y1 / h, 4)})-({round(x2 / w, 4)},{round(y2 / h, 4)})"
                )
        if not lines:
            return "（YOLO 未偵測到任何物件；請依原圖做一般職安觀察。）"
        return "\n".join(["[YOLO 偵測清單 — 請逐項對照圖片與法規]"] + lines)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def augment(src: Path, dst: Path, rng: random.Random):
    # 手機照片的 EXIF 方向要先轉正：cv2.imread（YOLO）會依 EXIF 旋轉，存檔後 EXIF 又會掉，
    # 不轉正的話增強圖會永久躺著，跟 YOLO 座標對不上。
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = img.size
    l, t = rng.uniform(0, AUG_MAX_CROP) * w, rng.uniform(0, AUG_MAX_CROP) * h
    r, b = w - rng.uniform(0, AUG_MAX_CROP) * w, h - rng.uniform(0, AUG_MAX_CROP) * h
    img = img.crop((int(l), int(t), int(r), int(b)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    img.thumbnail((AUG_SAVE_MAX_SIDE, AUG_SAVE_MAX_SIDE))
    img.save(dst, "JPEG", quality=rng.randint(70, 92))


def ext_label(label_path: Path, names):
    """回傳 response dict；無法判定（沒有人、或 PPE 標註不完整）回 None。"""
    cls = Counter()
    for line in label_path.read_text().splitlines():
        if line.strip():
            cls[names[int(line.split()[0])]] += 1
    findings = []
    if cls["no-helmet"]:
        findings.append({"item": EXT_ITEM, "result": EXT_NO_HELMET})
    if cls["no-vest"]:
        findings.append({"item": EXT_ITEM, "result": EXT_NO_VEST})
    if findings:
        return {"compliant": False, "findings": findings}
    if cls["person"] and cls["helmet"] and cls["vest"]:
        return {"compliant": True, "findings": []}
    return None


def _is_violation(v):
    return v not in (None, "", {}, []) and not (isinstance(v, float) and v != v)


def load_cs10k():
    """讀 CS10k 的 parquet，圖片解到 CS10K_IMG_DIR，回傳 [(img_path, response_dict 或 None)]。"""
    import io
    import pyarrow.parquet as pq
    files = sorted(CS10K_DIR.rglob("*.parquet"))
    CS10K_IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for fp in files:
        for row in pq.read_table(fp).to_pylist():
            ip = CS10K_IMG_DIR / f"{row['image_id']}.jpg"
            if not ip.exists():
                Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(ip, "JPEG", quality=92)
            findings = [dict(m) for k, m in CS10K_RULE_MAP.items() if _is_violation(row.get(k))]
            if findings:
                out.append((ip, {"compliant": False, "findings": findings}))
            elif _is_violation(row.get("rule_1_violation")):
                out.append((ip, None))  # 只違反 PPE：跳過
            else:
                out.append((ip, {"compliant": True, "findings": []}))
    return out


def load_web():
    """讀 web_labels.tsv，回傳 [(img_path, response_dict)]；skip 的不收。"""
    out = []
    for line in WEB_LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        img_id, label, findings = (line.split("	") + ["", ""])[:3]
        if label == "skip":
            continue
        resp = {"compliant": label == "compliant", "findings": []}
        if label == "violation":
            resp["findings"] = [dict(zip(("item", "result"), f.split("｜"))) for f in findings.split(";;")]
        out.append((WEB_DIRS[img_id[:2]] / f"{img_id}.jpg", resp))
    return out


def main():
    rng = random.Random(42)
    OUT_DIR.mkdir(exist_ok=True)
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(YOLO_CACHE.read_text(encoding="utf-8")) if YOLO_CACHE.exists() else {}
    yolo = Yolo()

    def block_for(path: Path):
        key = str(path)
        if key not in cache:
            cache[key] = yolo.block(path)
            if len(cache) % 100 == 0:
                print(f"  ...YOLO 已跑 {len(cache)} 張")
                YOLO_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return cache[key]

    def example(path: Path, resp: dict, source: str):
        return {"image": str(path), "prompt": f"{PROMPT_PREFIX}\n\n{block_for(path)}",
                "response": json.dumps(resp, ensure_ascii=False), "source": source}

    # 1. real
    real = read_jsonl(SRC_DIR / "train.jsonl")
    val = read_jsonl(SRC_DIR / "val.jsonl")
    val_imgs = {e["image"] for e in val}
    for e in real:
        e["source"] = "real"

    # 2. aug：每張 train 原圖只增強一次（過採樣的重複列不重複增強）
    uniq = {}
    for e in real:
        uniq.setdefault(e["image"], e)
    assert not (set(uniq) & val_imgs), "train/val 圖片重疊"
    aug = []
    for i, (img, e) in enumerate(sorted(uniq.items())):
        for k in range(AUG_PER_IMAGE):
            dst = AUG_DIR / f"{Path(img).stem}__aug{k}.jpg"
            if not dst.exists():
                augment(Path(img), dst, rng)
            aug.append(example(dst, json.loads(e["response"]), "aug"))
        if (i + 1) % 50 == 0:
            print(f"[aug] {i + 1}/{len(uniq)}")

    # 3. cs10k
    cs = []
    if CS10K_DIR.exists() and any(CS10K_DIR.rglob("*.parquet")):
        rows = load_cs10k()
        viol = [r for r in rows if r[1] and not r[1]["compliant"]]
        comp = [r for r in rows if r[1] and r[1]["compliant"]]
        rng.shuffle(viol)
        rng.shuffle(comp)
        picked = viol[:CS10K_MAX_VIOLATION] + comp[:CS10K_MAX_COMPLIANT]
        print(f"[cs10k] 共 {len(rows)} 張：違規 {len(viol)} / 合規 {len(comp)} / 只違反PPE跳過 "
              f"{sum(r[1] is None for r in rows)}，取 {len(picked)}")
        for i, (ip, resp) in enumerate(picked):
            cs.append(example(ip, resp, "cs10k"))
            if (i + 1) % 100 == 0:
                print(f"[cs10k] {i + 1}/{len(picked)}")
    else:
        print(f"[cs10k] 找不到 {CS10K_DIR} 的 parquet，略過（需先 hf auth login 並在網頁同意條款）")

    # 5. web：原圖 + 增強，缺失過採樣
    web = []
    if WEB_LABELS.exists():
        for ip, resp in load_web():
            paths = [ip]
            for k in range(AUG_PER_IMAGE):
                dst = AUG_DIR / f"{ip.stem}__aug{k}.jpg"
                if not dst.exists():
                    augment(ip, dst, rng)
                paths.append(dst)
            reps = 1 if resp["compliant"] else TRAIN_VIOLATION_OVERSAMPLE
            for path in paths:
                web += [example(path, resp, "web")] * reps
        print(f"[web] {len(web)} 筆")

    # 4. ppe（預設關閉）
    ext = []
    if USE_PPE_EXT and EXT_DIR.exists():
        import yaml
        names = yaml.safe_load((EXT_DIR / "data.yaml").read_text())["names"]
        viol, comp = [], []
        for split in ("train", "valid", "test"):
            for lp in sorted((EXT_DIR / split / "labels").glob("*.txt")):
                resp = ext_label(lp, names)
                if resp is None:
                    continue
                ip = next((lp.parent.parent / "images" / (lp.stem + s) for s in (".jpg", ".jpeg", ".png")
                           if (lp.parent.parent / "images" / (lp.stem + s)).exists()), None)
                if ip:
                    (comp if resp["compliant"] else viol).append((ip, resp))
        rng.shuffle(viol)
        rng.shuffle(comp)
        picked = viol[:EXT_MAX_VIOLATION] + comp[:EXT_MAX_COMPLIANT]
        print(f"[ext] 可用 缺失 {len(viol)} / 合規 {len(comp)}，取 {len(picked)}")
        for i, (ip, resp) in enumerate(picked):
            ext.append(example(ip, resp, "ext"))
            if (i + 1) % 100 == 0:
                print(f"[ext] {i + 1}/{len(picked)}")


    YOLO_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    train = real + aug + cs + web + ext
    rng.shuffle(train)
    with open(OUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for e in train:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    shutil.copyfile(SRC_DIR / "val.jsonl", OUT_DIR / "val.jsonl")

    def summarize(rows):
        resp = [json.loads(e["response"]) for e in rows]
        items = Counter(f"{f['item']}｜{f['result']}" for r in resp for f in r["findings"])
        return {"rows": len(rows), "unique_images": len({e["image"] for e in rows}),
                "compliant": sum(r["compliant"] for r in resp),
                "violation": sum(not r["compliant"] for r in resp),
                "top_findings": items.most_common(15)}

    stats = {"train_total": summarize(train), "val": summarize(val),
             **{f"train_{s}": summarize([e for e in train if e["source"] == s]) for s in ("real", "aug", "cs10k", "web", "ext")}}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in stats.items():
        print(f"{k:12s} rows={v['rows']:5d} 圖={v['unique_images']:5d} 合規={v['compliant']:5d} 缺失={v['violation']:5d}")


if __name__ == "__main__":
    main()
