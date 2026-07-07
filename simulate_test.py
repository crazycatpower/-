# -*- coding: utf-8 -*-
"""
simulate_test.py — 仿真正常 LINE 使用者操作流程
  1. 以 TEST.jpg 執行圖像分析（YOLO + LLM）
  2. 將結果寫入 PostgreSQL（audit_records）
  3. 模擬使用者查詢稽核紀錄、統計、案例學習
用法: python simulate_test.py [--image TEST.jpg] [--rounds 3]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── 導入主系統模組 ─────────────────────────────────────────────────────────
from app import (
    DB_AVAILABLE,
    DEFAULT_ANALYZE_PROMPT,
    MODE_IMAGE_ANALYSIS,
    analyze_image_bytes,
    format_audit_records_reply,
    get_case_study_stats,
    reply_case_study,
    save_audit_record,
    init_audit_db,
)

# 虛擬 LINE 使用者狀態（不需要真實 LINE 連線）
FAKE_SOURCE_KEY = "user:simulate_test_user"
FAKE_STATE: dict = {
    "mode": MODE_IMAGE_ANALYSIS,
    "prompt": DEFAULT_ANALYZE_PROMPT,
    "page_items": [],
    "page_index": 0,
    "page_query": "",
    "page_mode": "",
    "kb_pending": None,
}

SEP = "=" * 62


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def show_json(obj: dict, max_len: int = 400) -> None:
    raw = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(raw) > max_len:
        raw = raw[:max_len] + "\n  ..."
    print(raw)


def run_image_round(image_path: Path, round_idx: int, prompt: str) -> dict | None:
    section(f"Round {round_idx}：圖像分析 — {image_path.name}")
    print(f"分析提示：{prompt}")
    print("執行中…")

    t0 = time.perf_counter()
    image_bytes = image_path.read_bytes()

    try:
        result = analyze_image_bytes(
            image_bytes,
            prompt=prompt,
            use_yolo=True,
            filename_stem=f"sim_r{round_idx}",
        )
    except Exception as exc:
        print(f"[ERROR] analyze_image_bytes 失敗：{exc}")
        return None

    elapsed = time.perf_counter() - t0
    det_count = len(result.get("detections") or [])
    scene = result.get("scene_analysis") or {}

    print(f"耗時        : {elapsed:.1f}s")
    print(f"偵測物件數  : {det_count}")
    print(f"LLM 摘要    : {str(scene.get('summary') or scene.get('raw_text') or '')[:300]}")

    risks = scene.get("risks") or []
    if risks:
        print("風險列表    :")
        for r in risks[:5]:
            print(f"  • {r}")

    return result


def save_round(result: dict, round_idx: int, prompt: str) -> int | None:
    if not DB_AVAILABLE:
        print("[SKIP] DB 未連線，跳過儲存。")
        return None

    image_bytes = result.get("input_image_bytes") or b""
    try:
        record = save_audit_record(
            source="simulate_test",
            source_key=FAKE_SOURCE_KEY,
            prompt=prompt,
            original_image_data=image_bytes,
            result=result,
            analysis_mode="line image",
        )
        audit_id = record.get("id")
        print(f"已寫入 DB   : audit_records id=#{audit_id}")
        return audit_id
    except Exception as exc:
        print(f"[ERROR] 儲存 DB 失敗：{exc}")
        return None


def sim_audit_record_query(today_str: str) -> None:
    section("模擬使用者指令：稽核紀錄查詢")

    queries = [
        today_str[:7],            # 本月 (YYYY-MM)
        today_str,                # 今日 (YYYY-MM-DD)
        "統計",
    ]

    for q in queries:
        print(f"\n> 使用者輸入：「{q}」")
        state_copy = dict(FAKE_STATE)
        reply = format_audit_records_reply(q, FAKE_SOURCE_KEY, state_copy)
        print(reply[:1200])
        print("—" * 40)


def sim_case_study() -> None:
    section("模擬使用者指令：案例學習")
    if not DB_AVAILABLE:
        print("[SKIP] DB 未連線。")
        return

    for cmd in ["概覽", "本週", "最常見"]:
        print(f"\n> 使用者輸入：「{cmd}」")
        reply = reply_case_study(cmd)
        print(reply[:1500])
        print("—" * 40)


def sim_case_stats() -> None:
    section("原始案例統計數據（30 日）")
    if not DB_AVAILABLE:
        print("[SKIP] DB 未連線。")
        return

    stats = get_case_study_stats(30)
    ov = stats.get("overview") or {}
    print(f"稽核次數  : {ov.get('total') or 0}")
    print(f"平均偵測  : {ov.get('avg_det') or 0}")
    print(f"最高偵測  : {ov.get('max_det') or 0}")

    kw = [(k, c) for k, c in stats.get("kw_counts", []) if int(c or 0) > 0]
    if kw:
        print("常見違規  :")
        for k, c in kw[:7]:
            bar = "█" * min(int(c), 10)
            print(f"  {k}：{c} 次 {bar}")

    trend = stats.get("trend") or []
    if trend:
        print("近期趨勢  :")
        for row in trend[:7]:
            print(f"  {row['day']}：{row['cnt']} 次 / {row.get('detections') or 0} 項")


def main() -> None:
    parser = argparse.ArgumentParser(description="仿真 LINE 使用者完整操作流程測試")
    parser.add_argument("--image", default="TEST.jpg", help="測試圖片路徑（預設 TEST.jpg）")
    parser.add_argument("--rounds", type=int, default=3, help="重複分析次數（預設 3）")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = Path(__file__).parent / image_path
    if not image_path.exists():
        print(f"[ERROR] 找不到圖片：{image_path}")
        sys.exit(1)

    print(f"\n{'#'*62}")
    print("  AI 職安稽核系統 — 仿真測試")
    print(f"  圖片  : {image_path}")
    print(f"  輪次  : {args.rounds}")
    print(f"  DB    : {'已連線' if DB_AVAILABLE else '未連線（本地模式）'}")
    print(f"{'#'*62}")

    # 初始化 DB（幂等）
    init_audit_db()

    # ── 分析提示詞清單（模擬不同使用者需求） ─────────────────────────────
    prompts = [
        DEFAULT_ANALYZE_PROMPT,
        "請分析這張照片中的風險、異常與建議處置方式。",
        "請辨識畫面中的交通標誌或安全標示，說明其意義與注意事項。",
    ]

    audit_ids: list[int] = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── 主迴圈：多輪圖像分析 ──────────────────────────────────────────────
    for i in range(1, args.rounds + 1):
        prompt = prompts[(i - 1) % len(prompts)]
        result = run_image_round(image_path, i, prompt)
        if result is None:
            print(f"[WARN] Round {i} 分析失敗，跳過。")
            continue

        audit_id = save_round(result, i, prompt)
        if audit_id:
            audit_ids.append(audit_id)

        # 輪次之間稍作間隔，避免時間戳衝突
        if i < args.rounds:
            time.sleep(1.1)

    # ── 查詢模擬 ──────────────────────────────────────────────────────────
    sim_audit_record_query(today_str)
    sim_case_study()
    sim_case_stats()

    # ── 最終摘要 ──────────────────────────────────────────────────────────
    section("測試完成摘要")
    print(f"完成分析輪次 : {args.rounds}")
    print(f"成功寫入 DB  : {len(audit_ids)} 筆  {audit_ids}")
    print(f"測試時間     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    main()
