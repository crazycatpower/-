# -*- coding: utf-8 -*-
"""
不微調 VLM、在 v10 外面加東西的實驗：先算好每張圖的特徵，存在 posthoc/，再由 posthoc_eval.py 組合與評分。

  python posthoc_features.py vlm      # dev 照片的 v10 缺失機率（val57 / ext113 直接讀 results/ 的評測結果）
  python posthoc_features.py siglip   # 所有照片的 SigLIP 圖片向量（相似照片投票用）
  python posthoc_features.py owl      # dev / val57 / ext113 的 OWLv2 零樣本偵測分數（護欄、安全網、護蓋…）

照片分四組：
  pool   已標註的訓練照片（v10 train 與 finetune_data_fall train 的原圖），當作相似照片投票的資料庫
  dev    v10 dev（48 張）＋ 墜落 dev（finetune_data_fall/dev.jsonl），用來決定合併方式與門檻
  val57、ext113  測試集，只用來報分數
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import audit_vlm_common as C

OUT = C.BASE_DIR / "posthoc"
V10 = C.BASE_DIR / "gemma3n-e2b-audit-lora-v10" / "best"
SIGLIP_ID = "google/siglip-so400m-patch14-384"
OWL_ID = "google/owlv2-base-patch16-ensemble"
OWL_QUERIES = [
    "guardrail", "safety railing around an opening", "safety net", "stair handrail", "cover plate over a floor opening",
    "open hole in the floor", "unprotected edge of a building", "worker wearing a safety harness", "ladder", "scaffold",
    "stair tower", "aerial work platform",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_violation(response):
    return not json.loads(response)["compliant"]


def sets():
    """{組名: [{"key", "path", "label"(True=缺失), "item", "fall"(bool)}...]}"""
    fall_dev_groups = {r["group"] for r in read_jsonl(C.BASE_DIR / "finetune_data_fall" / "dev.jsonl")}
    fall_train_groups = {r["group"] for r in read_jsonl(C.BASE_DIR / "finetune_data_fall" / "train.jsonl")}
    fall_groups = fall_dev_groups | fall_train_groups

    def from_rows(rows):
        out, seen = [], set()
        for r in rows:
            if r["version"] != 0 or r["image"] in seen:
                continue
            seen.add(r["image"])
            out.append({"key": r["image"], "path": str(C.BASE_DIR / r["image"]), "label": is_violation(r["response"]),
                        "item": r["item"], "fall": r["group"] in fall_groups, "prompt": r["prompt"],
                        "response": r["response"]})
        return out

    pool = from_rows(read_jsonl(C.BASE_DIR / "finetune_data_v10" / "train.jsonl")
                     + read_jsonl(C.BASE_DIR / "finetune_data_fall" / "train.jsonl"))
    dev = from_rows(read_jsonl(C.BASE_DIR / "finetune_data_v10" / "dev.jsonl")
                    + read_jsonl(C.BASE_DIR / "finetune_data_fall" / "dev.jsonl"))
    from compare_fall import FALL

    tests = {}
    for name, img_dir in (("val57", C.IMG_DIR), ("ext113", C.EXT_DIR)):
        rows = json.load(open(C.BASE_DIR / "results" / f"{name}_gemma3n-e2b-audit-lora-v10_best.json", encoding="utf-8"))["rows"]
        tests[name] = [{"key": r["image"], "path": str(img_dir / r["image"]), "label": r["gt_violation"],
                        "item": r["item"], "fall": Path(r["image"]).name in FALL, "p_vlm": r["p_violation"],
                        "pred_vlm": r["pred_violation"]} for r in rows]
    return {"pool": pool, "dev": dev, **tests}


def run_vlm(all_sets):
    """dev 照片的 v10 缺失機率：teacher forcing 下答案 true/false 那一格選 false 的機率，與生成時的 p_violation 相同。"""
    model, processor = C.load_model(adapter=str(V10))
    model.eval()
    dec_pos, true_id, false_id = C.decision_token_ids(processor.tokenizer)
    out = {}
    with torch.no_grad():
        for r in all_sets["dev"]:
            batch, labels = C.encode(processor, C.load_image(r["path"]), r["prompt"], r["response"])
            logits, _ = C.answer_logits(model, batch, labels)
            out[r["key"]] = torch.softmax(logits[dec_pos, [true_id, false_id]], 0)[1].item()
    (OUT / "dev_vlm.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("dev 照片", len(out))


def run_siglip(all_sets):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(SIGLIP_ID, torch_dtype=torch.float16).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    keys, vecs = [], []
    for name, rows in all_sets.items():
        for r in rows:
            img = C.load_image(r["path"])
            with torch.no_grad():
                inputs = proc(images=[img], return_tensors="pt").to("cuda")
                v = model.get_image_features(pixel_values=inputs["pixel_values"].half())
                v = getattr(v, "pooler_output", v)
            vecs.append(torch.nn.functional.normalize(v.float(), dim=-1)[0].cpu().numpy())
            keys.append(f"{name}|{r['key']}")
        print(name, len(rows), flush=True)
    np.savez(OUT / "siglip.npz", keys=np.array(keys), vecs=np.stack(vecs))


def run_owl(all_sets):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model = Owlv2ForObjectDetection.from_pretrained(OWL_ID).to("cuda").eval()
    proc = Owlv2Processor.from_pretrained(OWL_ID)
    out = {}
    for name in ("dev", "val57", "ext113"):
        for r in all_sets[name]:
            img = C.load_image(r["path"])
            with torch.no_grad():
                inputs = proc(text=[OWL_QUERIES], images=img, return_tensors="pt").to("cuda")
                logits = model(**inputs).logits[0]  # [boxes, queries]
            out[f"{name}|{r['key']}"] = torch.sigmoid(logits).max(0).values.cpu().tolist()
        print(name, len(all_sets[name]), flush=True)
    (OUT / "owl.json").write_text(json.dumps({"queries": OWL_QUERIES, "scores": out}, ensure_ascii=False), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    all_sets = sets()
    meta = {k: [{x: r[x] for x in ("key", "path", "label", "item", "fall", "p_vlm", "pred_vlm") if x in r} for r in v]
            for k, v in all_sets.items()}
    (OUT / "sets.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    {"vlm": run_vlm, "siglip": run_siglip, "owl": run_owl}[sys.argv[1]](all_sets)


if __name__ == "__main__":
    main()
