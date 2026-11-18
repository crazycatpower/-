# -*- coding: utf-8 -*-
"""
把 v10 稽核 LoRA 的輸出（{"compliant", "findings":[{"item","result"}]}）轉成 safety_audit_system.py 要的
場景分析 JSON（categories 依稽核表代碼分類、summary、answer…），讓 llm_server.py 的稽核模式能直接接主程式。

v10 只學過短 prompt（PROMPT_PREFIX + YOLO 偵測清單）與上述簡短格式；主程式的長 prompt（54 條清單＋法規）它沒看過，
所以稽核模式不使用主程式送來的 prompt，只取原圖，自己跑 YOLO 組出訓練時的 prompt。
缺失項目的文字是稽核表條文（或答案表的寫法），用字元 bigram 相似度對到 audit_checklist 的官方代碼。
"""
import json

from audit_checklist import CHECKLIST_1_TO_6, CODE_TO_EXPECTED_CATEGORY, OTHER_REFERENCE_ITEMS

_ALL_ITEMS = [ci for items in CHECKLIST_1_TO_6.values() for ci in items] + \
             [ci for items in OTHER_REFERENCE_ITEMS.values() for ci in items]


def _bigrams(s):
    s = "".join(ch for ch in s if not ch.isspace() and ch not in "，。、：:；;（）()□")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def code_for(item, result):
    """缺失（項目文字＋缺失描述）→ 最相近的官方代碼；相似度太低回 None。"""
    query = _bigrams(f"{item}{result}")
    if not query:
        return None
    best, best_score = None, 0.0
    for ci in _ALL_ITEMS:
        target = _bigrams(ci.text)
        score = len(query & target) / len(query)
        if score > best_score:
            best, best_score = ci.item, score
    return best if best_score >= 0.3 else None


def to_scene_json(text, p_violation=None):
    """v10 的原始輸出 → 主程式的場景分析 dict。輸出壞掉時回傳只有 summary 的 dict（主程式會照常補齊欄位）。"""
    t = text.strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        raw = json.loads(t[s:e + 1])
    except Exception:  # noqa: BLE001
        return {"summary": "稽核模型輸出無法解析，請人工確認。", "answer": t[:300]}

    categories, unmapped = {}, []
    for f in raw.get("findings") or []:
        item, result = str(f.get("item") or ""), str(f.get("result") or "")
        code = code_for(item, result)
        if code is None:
            unmapped.append(f"{item}{result}")
            continue
        categories.setdefault(CODE_TO_EXPECTED_CATEGORY[code], []).append(
            {"item": code, "result": "×", "description": f"{item}{result}", "location": ""})

    n = sum(len(v) for v in categories.values()) + len(unmapped)
    conf = "" if p_violation is None else f"（模型判定缺失機率 {p_violation:.2f}）"
    if n == 0:
        summary = f"未發現明確缺失{conf}。"
    else:
        listed = "；".join([it["description"] for v in categories.values() for it in v] + unmapped)
        summary = f"發現 {n} 項缺失{conf}：{listed}"
    return {"summary": summary, "answer": summary, "categories": categories,
            "audit_model": {"compliant": not n, "p_violation": p_violation, "unmapped_findings": unmapped}}
