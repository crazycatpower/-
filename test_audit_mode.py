# -*- coding: utf-8 -*-
"""
端到端檢查 llm_server.py 稽核模式：用主程式的 ask_llm_scene_analysis（完整 prompt、後處理、54 條清單補齊）
呼叫稽核模式的服務，看最後主程式拿到的結果是否和 test_val57.py --threshold 的判定一致、格式是否完整。

先啟動：LORA_AUDIT_MODE=1 LORA_ADAPTER_DIR=gemma3n-e2b-audit-lora-v10/best LORA_PORT=8002 python llm_server.py
再執行：python test_audit_mode.py [張數]
"""
import json
import os
import sys

from openai import OpenAI

import audit_vlm_common as C
import safety_audit_system as S

URL = "http://127.0.0.1:8002/v1"
os.environ["LLM_BASE_URL"] = URL  # import 時 .env 會設成 Ollama，這裡覆蓋回稽核模式的服務
os.environ["LLM_FORCE_JSON"] = "1"

n = int(sys.argv[1]) if len(sys.argv) > 1 else 57
ref = {r["image"]: r for r in json.load(open(C.BASE_DIR / "results" / "val57_gemma3n-e2b-audit-lora-v10_best_thr0.64.json",
                                             encoding="utf-8"))["rows"]}
client = OpenAI(api_key="local", base_url=URL)
same = correct = complete = 0
rows = list(ref.values())[:n]
for i, r in enumerate(rows, 1):
    out = S.ask_llm_scene_analysis(client, "gemma3n-e2b-audit-lora", C.IMG_DIR / r["image"], "請稽核這張工地照片",
                                   [], supports_vision=True, use_yolo=False)
    cats = out.get("categories") or {}
    defects = [it["item"] for items in cats.values() for it in items if it.get("result") in ("×", "△")]
    pred = bool(defects)
    n_items = sum(len(cats.get(c, [])) for c in S.CHECKLIST_1_TO_6)
    complete += n_items == 54 and all(out.get(k) for k in ("summary", "next_action", "answer", "regulation_refs"))
    same += pred == r["pred_violation"]
    correct += pred == r["gt_violation"]
    print(f"[{i}/{len(rows)}] {r['image'][:30]} 主程式判定={pred} 評測判定={r['pred_violation']} 答案={r['gt_violation']} 缺失代碼={defects}", flush=True)
print(f"\n與 test_val57 --threshold 0.64 判定一致 {same}/{len(rows)}；對答案 {correct}/{len(rows)}；"
      f"格式完整（54 條＋必要欄位）{complete}/{len(rows)}")
