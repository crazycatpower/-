# -*- coding: utf-8 -*-
"""
investigation_report.py — AI 職安稽核調查報告產生器
參考 benchmark.py 風格，從 PostgreSQL 讀取 audit_records 生成 HTML 報告。
用法: python investigation_report.py [--days 30] [--limit 50] [--output report.html]
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── 資料庫存取（獨立查詢，不需要 Flask/LINE 環境） ──────────────────────────
_DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
DB_AVAILABLE = bool(_DATABASE_URL)

_KEYWORDS = ["安全帽", "安全帶", "鷹架", "護欄", "反光背心", "安全鞋", "電氣", "火災", "墜落", "感電"]
_HIGH_RISK_KW = {"墜落", "感電", "觸電", "火災", "爆炸", "倒塌", "中毒", "高空", "缺氧", "窒息"}
_MED_RISK_KW  = {"未佩戴", "未穿", "缺少", "違規", "no-helmet", "no-vest", "未配戴", "未戴"}


def _conn():
    import psycopg
    from psycopg.rows import dict_row
    return psycopg.connect(_DATABASE_URL, row_factory=dict_row)


# ── 資料收集 ──────────────────────────────────────────────────────────────────

def gather_overview(since: str) -> dict:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                           AS total,
                    ROUND(AVG(detection_count)::numeric, 1) AS avg_det,
                    MAX(detection_count)               AS max_det,
                    SUM(detection_count)               AS total_det,
                    MIN(SUBSTR(created_at,1,10))       AS earliest,
                    MAX(SUBSTR(created_at,1,10))       AS latest
                FROM audit_records
                WHERE SUBSTR(created_at,1,10) >= %s
            """, (since,))
            return cur.fetchone() or {}


def gather_keyword_hits(since: str) -> list[tuple[str, int]]:
    kw = _KEYWORDS
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT " + ", ".join(
                    f"SUM(CASE WHEN summary ILIKE %s THEN 1 ELSE 0 END) AS k{i}"
                    for i, _ in enumerate(kw)
                ) + " FROM audit_records WHERE SUBSTR(created_at,1,10) >= %s",
                [f"%{k}%" for k in kw] + [since],
            )
            row = cur.fetchone() or {}
    return sorted(
        [(kw[i], int(row.get(f"k{i}") or 0)) for i in range(len(kw))],
        key=lambda x: x[1], reverse=True,
    )


def gather_daily_trend(since: str) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT SUBSTR(created_at,1,10) AS day,
                       COUNT(*)               AS cnt,
                       SUM(detection_count)   AS detections,
                       MAX(detection_count)   AS peak_det
                FROM audit_records
                WHERE SUBSTR(created_at,1,10) >= %s
                GROUP BY day ORDER BY day DESC LIMIT 30
            """, (since,))
            return cur.fetchall()


def gather_source_breakdown(since: str) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source,
                       COUNT(*)             AS cnt,
                       SUM(detection_count) AS total_det,
                       ROUND(AVG(detection_count)::numeric,1) AS avg_det
                FROM audit_records
                WHERE SUBSTR(created_at,1,10) >= %s
                GROUP BY source ORDER BY cnt DESC
            """, (since,))
            return cur.fetchall()


def gather_records(since: str, limit: int) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, created_at, source, analysis_mode,
                       detection_count, summary, report_json
                FROM audit_records
                WHERE SUBSTR(created_at,1,10) >= %s
                ORDER BY created_at DESC LIMIT %s
            """, (since, limit))
            return cur.fetchall()


def infer_risk(summary: str, det_count: int) -> tuple[str, str]:
    """回傳 (等級文字, badge 色)。"""
    t = str(summary or "")
    if any(k in t for k in _HIGH_RISK_KW):
        return "高", "#ef4444"
    if det_count >= 3 or any(k in t for k in _MED_RISK_KW):
        return "中", "#f97316"
    if det_count >= 1:
        return "低", "#eab308"
    return "正常", "#22c55e"


def gather_all(days: int, limit: int) -> dict:
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"查詢起始日期 : {since}")
    overview  = gather_overview(since)
    keywords  = gather_keyword_hits(since)
    trend     = gather_daily_trend(since)
    sources   = gather_source_breakdown(since)
    records   = gather_records(since, limit)
    print(f"取得紀錄     : {len(records)} 筆（上限 {limit}）")
    return {
        "since": since,
        "days": days,
        "overview": overview,
        "keywords": keywords,
        "trend": trend,
        "sources": sources,
        "records": records,
    }


# ── badge 工具（同 benchmark.py） ─────────────────────────────────────────────

def _badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def _det_badge(n: int) -> str:
    color = (
        "#22c55e" if n == 0 else
        "#eab308" if n <= 2 else
        "#f97316" if n <= 4 else
        "#ef4444"
    )
    return _badge(str(n), color)


# ── HTML 報告產生器 ────────────────────────────────────────────────────────────

def generate_html(data: dict, out_path: Path) -> None:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ov       = data["overview"]
    total    = int(ov.get("total") or 0)
    days     = data["days"]

    # ── 總覽摘要列 ──────────────────────────────────────────
    overview_rows = f"""
        <tr><td>調查期間</td><td>{data['since']} ～ {ov.get('latest') or '—'}</td></tr>
        <tr><td>稽核總筆數</td><td><strong>{total}</strong> 筆</td></tr>
        <tr><td>偵測項目總計</td><td>{int(ov.get('total_det') or 0)} 項</td></tr>
        <tr><td>平均偵測 / 次</td><td>{ov.get('avg_det') or 0} 項</td></tr>
        <tr><td>單次最高偵測</td><td>{int(ov.get('max_det') or 0)} 項</td></tr>
    """

    # ── 來源分佈表 ──────────────────────────────────────────
    source_rows = ""
    for s in data["sources"]:
        pct = round(int(s["cnt"]) / total * 100, 1) if total else 0
        bar = "█" * int(pct / 5)
        source_rows += f"""
        <tr>
          <td><strong>{s['source'] or '（未知）'}</strong></td>
          <td>{s['cnt']} 筆</td>
          <td>{pct}% <span style="color:#6366f1;font-size:.8rem">{bar}</span></td>
          <td>{s['avg_det']} 項</td>
          <td>{int(s['total_det'] or 0)} 項</td>
        </tr>"""

    # ── 關鍵字命中表 ────────────────────────────────────────
    kw_rows = ""
    top_count = max((c for _, c in data["keywords"]), default=1) or 1
    for rank, (kw, cnt) in enumerate(data["keywords"], 1):
        if cnt == 0:
            continue
        bar_w = max(4, int(cnt / top_count * 120))
        risk_color = "#ef4444" if kw in _HIGH_RISK_KW else "#f97316" if kw in {"墜落","感電"} else "#6366f1"
        kw_rows += f"""
        <tr>
          <td style="color:#64748b">{rank}</td>
          <td><strong>{kw}</strong></td>
          <td>{cnt} 次</td>
          <td><div style="height:10px;width:{bar_w}px;background:{risk_color};border-radius:4px"></div></td>
        </tr>"""
    if not kw_rows:
        kw_rows = '<tr><td colspan="4" style="color:#94a3b8">無命中關鍵字</td></tr>'

    # ── 每日趨勢表 ──────────────────────────────────────────
    trend_rows = ""
    for row in data["trend"]:
        det = int(row.get("detections") or 0)
        peak = int(row.get("peak_det") or 0)
        trend_rows += f"""
        <tr>
          <td>{row['day']}</td>
          <td>{row['cnt']} 筆</td>
          <td>{_det_badge(det)}</td>
          <td>{peak}</td>
        </tr>"""
    if not trend_rows:
        trend_rows = '<tr><td colspan="4" style="color:#94a3b8">無資料</td></tr>'

    # ── 個案紀錄卡片 ────────────────────────────────────────
    record_cards = ""
    for rec in data["records"]:
        det   = int(rec.get("detection_count") or 0)
        risk, risk_color = infer_risk(rec.get("summary") or "", det)
        ts    = str(rec.get("created_at") or "")[:16]
        src   = rec.get("source") or "—"
        mode  = rec.get("analysis_mode") or "—"
        rid   = rec.get("id") or "?"

        summary = str(rec.get("summary") or "").strip().replace("<", "&lt;") or "（無摘要）"
        if len(summary) > 500:
            summary = summary[:500] + "…"

        # 從 report_json 取 risks
        rj = rec.get("report_json") or {}
        if isinstance(rj, str):
            try:
                rj = json.loads(rj)
            except Exception:
                rj = {}
        scene = rj.get("scene_analysis") or rj
        raw_risks = scene.get("risks") or []
        risks_html = ""
        if raw_risks:
            items = "".join(
                f'<li style="margin:.15rem 0">{str(r)[:120].replace("<","&lt;")}</li>'
                for r in raw_risks[:6]
            )
            risks_html = f'<ul style="margin:.4rem 0 0 1rem;font-size:.8rem;color:#374151">{items}</ul>'

        next_action = str(scene.get("next_action") or "").strip()[:300].replace("<", "&lt;")

        stats = (
            f'時間：<strong>{ts}</strong> | '
            f'來源：<strong>{src}</strong> | '
            f'模式：<strong>{mode}</strong> | '
            f'偵測數：{_det_badge(det)} | '
            f'風險：{_badge(risk, risk_color)}'
        )

        record_cards += f"""
        <div class="card">
          <div class="card-header" style="background:{risk_color}">
            #{rid} &nbsp;|&nbsp; {ts} &nbsp;|&nbsp; {risk}風險
          </div>
          <div class="stats">{stats}</div>
          <div class="io-grid">
            <div class="io-box">
              <div class="io-label">摘要</div>
              <pre>{summary}</pre>
              {risks_html}
            </div>
            <div class="io-box">
              <div class="io-label">建議改善措施</div>
              <pre>{next_action or "（無建議）"}</pre>
            </div>
          </div>
        </div>"""

    if not record_cards:
        record_cards = '<p style="color:#94a3b8;padding:1rem">此期間無稽核紀錄。</p>'

    # ── 組合 HTML ───────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 職安稽核調查報告 — {run_time}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .25rem; }}
  .meta {{ color: #64748b; font-size: .875rem; margin-bottom: 2rem; }}
  h2.section {{ font-size: 1.15rem; margin: 2rem 0 .75rem; color: #334155; border-left: 4px solid #6366f1; padding-left: .6rem; }}
  h2.rec-title {{ font-size: 1rem; margin: 2rem 0 .75rem; color: #475569; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  th {{ background: #6366f1; color: #fff; padding: .65rem 1rem; text-align: left; font-size: .85rem; }}
  td {{ padding: .6rem 1rem; border-bottom: 1px solid #e2e8f0; font-size: .9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ border-radius: 999px; padding: .15rem .55rem; font-size: .75rem; color: #fff; font-weight: 600; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 1rem; overflow: hidden; }}
  .card-header {{ color: #fff; padding: .5rem 1rem; font-weight: 600; font-size: .9rem; }}
  .stats {{ padding: .6rem 1rem; font-size: .82rem; color: #475569; border-bottom: 1px solid #e2e8f0; }}
  .io-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #e2e8f0; }}
  .io-box {{ background: #fff; padding: .75rem 1rem; }}
  .io-label {{ font-size: .75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: .35rem; }}
  pre {{ font-family: "Cascadia Code", "Consolas", monospace; font-size: .8rem; white-space: pre-wrap; word-break: break-word; max-height: 260px; overflow-y: auto; color: #334155; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 700px) {{ .io-grid, .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>AI 職安稽核調查報告</h1>
<div class="meta">
  產生時間：{run_time} ｜ 調查期間：近 {days} 日 ｜ 稽核筆數：{total} 筆 ｜ 顯示紀錄：{len(data['records'])} 筆
</div>

<h2 class="section">整體摘要</h2>
<table>
  <thead><tr><th>指標</th><th>數值</th></tr></thead>
  <tbody>{overview_rows}</tbody>
</table>

<div class="two-col" style="margin-top:1.5rem">

  <div>
    <h2 class="section">來源分佈</h2>
    <table>
      <thead><tr><th>來源</th><th>筆數</th><th>佔比</th><th>平均偵測</th><th>偵測總項</th></tr></thead>
      <tbody>{source_rows or '<tr><td colspan="5" style="color:#94a3b8">無資料</td></tr>'}</tbody>
    </table>
  </div>

  <div>
    <h2 class="section">違規關鍵字命中排行</h2>
    <table>
      <thead><tr><th>#</th><th>關鍵字</th><th>命中次數</th><th>分佈</th></tr></thead>
      <tbody>{kw_rows}</tbody>
    </table>
  </div>

</div>

<h2 class="section">每日稽核趨勢</h2>
<table>
  <thead><tr><th>日期</th><th>稽核次數</th><th>偵測項目總計</th><th>單日最高偵測</th></tr></thead>
  <tbody>{trend_rows}</tbody>
</table>

<h2 class="section">個案稽核紀錄</h2>
{record_cards}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI 職安稽核調查報告產生器")
    parser.add_argument("--days",   type=int, default=30,   help="調查天數（預設 30）")
    parser.add_argument("--limit",  type=int, default=50,   help="個案紀錄上限（預設 50）")
    parser.add_argument("--output", default=None,           help="輸出 HTML 路徑（預設自動命名）")
    parser.add_argument("--json",   default=None,           help="同時輸出原始 JSON")
    args = parser.parse_args()

    if not DB_AVAILABLE:
        print("ERROR：請在 .env 設定 DATABASE_URL 後再執行。")
        return

    print(f"資料庫  : {_DATABASE_URL[:40]}…")
    print(f"查詢範圍: 近 {args.days} 日 / 個案上限 {args.limit} 筆")

    data = gather_all(args.days, args.limit)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    html_path = Path(args.output) if args.output else out_dir / f"investigation_{ts}.html"
    generate_html(data, html_path)
    print(f"\n報告已儲存 → {html_path.resolve()}")

    json_path = Path(args.json) if args.json else out_dir / f"investigation_{ts}.json"
    # report_json (JSONB) 欄位不可序列化，先轉字串
    safe_data = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    json_path.write_text(json.dumps(safe_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原始資料 → {json_path.resolve()}")


if __name__ == "__main__":
    main()
