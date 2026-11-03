# -*- coding: utf-8 -*-
"""
用 test_images/ 裡 6 張從未在訓練/驗證集出現過的真實照片，
接上 app 真實的 YOLO 偵測流程（跟 benchmark.py 一樣），測試 v1 微調模型
在真正未知照片上的實際表現。
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from peft import PeftModel

from benchmark import run_yolo

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test_images"
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


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, **_LOAD_KWARGS)
    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map={"": 0}, torch_dtype=torch.bfloat16, **_LOAD_KWARGS,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    images = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png"))
    results = []
    for img_path in images:
        detections, detection_block = run_yolo(img_path)
        img = Image.open(img_path).convert("RGB")
        prompt_text = f"請分析這張施工現場照片的職安合規性。\n\n{detection_block}"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[[img]], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        out_text = processor.decode(gen, skip_special_tokens=True)

        print("=" * 70)
        print(img_path.name)
        print(f"  YOLO 偵測到 {len(detections)} 個物件")
        print("  模型輸出:", out_text[:500])
        results.append({"image": img_path.name, "n_detections": len(detections),
                         "output": out_text})

    with open(BASE_DIR / "test_real_flow_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n存到 test_real_flow_result.json")


if __name__ == "__main__":
    main()
