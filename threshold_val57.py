# -*- coding: utf-8 -*-
"""
讀模型在 `{"compliant": ` 後面選 false（＝違規）的機率，看不同門檻下的精確度／召回率取捨。
不重新訓練，只做一次 forward：把模型固定的開頭 ```json\n{"compliant":  接在 prompt 後面，
取下一個 token 是 true 還是 false 的 softmax 機率。

注意：門檻是在 val57 上掃的，同一批資料不能拿來「選」門檻又報分數；這裡只畫取捨曲線。
用法：python threshold_val57.py gemma4-e2b-audit-lora-v6/final
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel

from benchmark import run_yolo
from test_100_real_flow_v6 import MODEL_ID, _LOAD_KWARGS, SYSTEM_PROMPT

BASE_DIR = Path(__file__).resolve().parent
VAL = BASE_DIR / "finetune_data" / "val.jsonl"
IMG_DIR = BASE_DIR / "uploads" / "UploadImg"
ANSWER_PREFIX = '```json\n{"compliant": '


def main():
    adapter = sys.argv[1]
    processor = AutoProcessor.from_pretrained(MODEL_ID, **_LOAD_KWARGS)
    base = Gemma3nForConditionalGeneration.from_pretrained(MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16, **_LOAD_KWARGS)
    model = PeftModel.from_pretrained(base, str(BASE_DIR / adapter)).eval()
    tok = processor.tokenizer
    t_id, f_id = tok.encode("true", add_special_tokens=False), tok.encode("false", add_special_tokens=False)
    assert len(t_id) == 1 and len(f_id) == 1, (t_id, f_id)
    t_id, f_id = t_id[0], f_id[0]

    rows = []
    for i, ex in enumerate(map(json.loads, open(VAL, encoding="utf-8")), 1):
        img_path = IMG_DIR / Path(ex["image"]).name
        _, block = run_yolo(img_path)
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": f"請分析這張施工現場照片的職安合規性。\n\n{block}"}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
        inputs = processor(text=text, images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        p = torch.softmax(logits[[t_id, f_id]], dim=0)[1].item()
        rows.append({"image": img_path.name, "gt_violation": not json.loads(ex["response"])["compliant"], "p_violation": round(p, 4)})
        print(f"[{i}/57] p_violation={p:.3f} gt={rows[-1]['gt_violation']}", flush=True)

    curve = []
    for th in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        tp = sum(r["p_violation"] >= th and r["gt_violation"] for r in rows)
        fp = sum(r["p_violation"] >= th and not r["gt_violation"] for r in rows)
        fn = sum(r["p_violation"] < th and r["gt_violation"] for r in rows)
        tn = len(rows) - tp - fp - fn
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        curve.append({"threshold": th, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": round((tp + tn) / len(rows), 3),
                      "precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / (P + R), 3) if P + R else 0})
        print(curve[-1], flush=True)
    out = BASE_DIR / f"threshold_val57_{Path(adapter).parent.name}.json"
    out.write_text(json.dumps({"adapter": adapter, "rows": rows, "curve": curve}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
