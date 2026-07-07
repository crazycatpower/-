# -*- coding: utf-8 -*-
"""
Combined benchmark report generator.
Merges regular benchmark JSON + tool-use benchmark JSON into one HTML.

Usage:
    python make_report.py
    python make_report.py --bench outputs/benchmark_*.json --tool outputs/toolbench_*.json
"""

import argparse
import base64
import json
import statistics
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_ORDER = [
    "gemma4:latest",
    "phi4:latest",
    "mistral:latest",
    "gemma3:latest",
    "qwen3.5:9b",
]
MODEL_COLORS = {
    "gemma4:latest":  "#6366f1",
    "phi4:latest":    "#0ea5e9",
    "mistral:latest": "#f97316",
    "gemma3:latest":  "#22c55e",
    "qwen3.5:9b":     "#a855f7",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_latest(glob: str) -> list[dict]:
    files = sorted(BASE_DIR.glob(glob))
    if not files:
        return []
    return json.loads(files[-1].read_text(encoding="utf-8"))


def avg(vals: list) -> float:
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 2) if vals else 0.0


# ── Summary stats from benchmark data ─────────────────────────────────────────

def bench_summary(bench: list[dict]) -> dict[str, dict]:
    """Returns {model: {text_avg_time, text_avg_tps, img_avg_time, img_avg_tps, vision, errors}}"""
    out: dict[str, dict] = {}
    models = list(dict.fromkeys(r["model"] for r in bench))
    for m in models:
        rows = [r for r in bench if r["model"] == m and not r["error"]]
        text_rows = [r for r in rows if r.get("category") == "text"]
        img_rows  = [r for r in rows if r.get("category") == "image"]
        errors    = sum(1 for r in bench if r["model"] == m and r["error"])
        out[m] = {
            "text_avg_time": avg([r["total_time"] for r in text_rows]),
            "text_avg_tps":  avg([r["tokens_per_sec"] for r in text_rows]),
            "img_avg_time":  avg([r["total_time"] for r in img_rows]),
            "img_avg_tps":   avg([r["tokens_per_sec"] for r in img_rows]),
            "vision": any(r.get("vision_used") for r in img_rows),
            "errors": errors,
        }
    return out


def tool_summary(tool: list[dict]) -> dict[str, dict]:
    """Returns {model: {calls_total, calls_avg, tool_support, avg_time, results}}"""
    out: dict[str, dict] = {}
    models = list(dict.fromkeys(r["model"] for r in tool))
    for m in models:
        rows = [r for r in tool if r["model"] == m and not r["error"]]
        calls_per_case = [len(r.get("tool_calls_made") or []) for r in rows]
        out[m] = {
            "calls_total":   sum(calls_per_case),
            "calls_avg":     round(avg(calls_per_case), 1),
            "tool_support":  sum(calls_per_case) > 0,
            "avg_time":      avg([r.get("total_time") for r in rows]),
            "cases":         rows,
        }
    return out


# ── HTML building blocks ───────────────────────────────────────────────────────

def _badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;border-radius:999px;padding:.15rem .55rem;font-size:.72rem;font-weight:700">{text}</span>'


def _bar(pct: float, color: str, width: int = 120) -> str:
    w = int(width * min(pct, 1.0))
    return (
        f'<div style="display:inline-block;background:#e2e8f0;border-radius:4px;'
        f'width:{width}px;height:10px;vertical-align:middle">'
        f'<div style="background:{color};width:{w}px;height:10px;border-radius:4px"></div></div>'
    )


def section_overview(bsum: dict, tsum: dict) -> str:
    """Big summary card per model."""
    rows = ""
    # normalise speed for bar chart (max among models)
    max_tps = max(max((v["text_avg_tps"] for v in bsum.values()), default=0), 1)
    max_tool = max(max((v["calls_total"] for v in tsum.values()), default=0), 1)

    for m in MODEL_ORDER:
        if m not in bsum:
            continue
        b = bsum[m]
        t = tsum.get(m, {})
        color = MODEL_COLORS.get(m, "#64748b")

        vision_badge = _badge("Vision", "#8b5cf6") if b["vision"] else _badge("Text only", "#94a3b8")
        tool_badge   = _badge("Tool Use ✓", "#22c55e") if t.get("tool_support") else _badge("Tool Use ✗", "#ef4444")

        speed_bar  = _bar(b["text_avg_tps"] / max_tps, color)
        tool_calls = t.get("calls_total", 0)
        tool_bar   = _bar(tool_calls / max_tool, "#22c55e" if tool_calls else "#ef4444")

        rows += f"""
        <tr>
          <td><span style="font-weight:700;color:{color}">{m}</span></td>
          <td>{vision_badge}</td>
          <td>{tool_badge}</td>
          <td>{b['text_avg_time']} s</td>
          <td>{speed_bar} {b['text_avg_tps']} tok/s</td>
          <td>{b['img_avg_time']} s</td>
          <td>{b['img_avg_tps']} tok/s</td>
          <td>{tool_bar} {tool_calls} 次</td>
          <td>{"—" if not b['errors'] else f"<span style='color:#ef4444'>{b['errors']}</span>"}</td>
        </tr>"""

    return f"""
    <h2 class="section">模型總覽</h2>
    <table>
      <thead>
        <tr>
          <th>模型</th><th>Vision</th><th>Tool Use</th>
          <th>文字平均耗時</th><th>文字速度</th>
          <th>圖片平均耗時</th><th>圖片速度</th>
          <th>工具呼叫次數</th><th>錯誤</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def section_capability_matrix(bsum: dict, tsum: dict) -> str:
    """Capability matrix with coloured cells."""
    rows = ""
    for m in MODEL_ORDER:
        if m not in bsum:
            continue
        b = bsum[m]
        t = tsum.get(m, {})
        color = MODEL_COLORS.get(m, "#64748b")

        def cell(ok: bool, label: str = "") -> str:
            if ok:
                return f'<td style="text-align:center;background:#dcfce7">{label or "✓"}</td>'
            return f'<td style="text-align:center;background:#fee2e2">{label or "✗"}</td>'

        rows += f"""
        <tr>
          <td><span style="font-weight:700;color:{color}">{m}</span></td>
          {cell(True, "✓")}
          {cell(b['vision'])}
          {cell(t.get('tool_support', False))}
          {cell(b['text_avg_tps'] > 5, f"{b['text_avg_tps']} t/s")}
          {cell(b['text_avg_time'] < 15, f"{b['text_avg_time']} s")}
          {cell(b['errors'] == 0, "穩定" if b['errors'] == 0 else f"{b['errors']} 錯")}
        </tr>"""

    return f"""
    <h2 class="section">能力矩陣</h2>
    <table>
      <thead>
        <tr>
          <th>模型</th><th>文字理解</th><th>視覺分析</th><th>工具呼叫</th>
          <th>速度 (&gt;5 t/s)</th><th>延遲 (&lt;15 s)</th><th>穩定性</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def section_tool_detail(tsum: dict) -> str:
    """Per-model per-case tool use detail."""
    cards = ""
    for m in MODEL_ORDER:
        t = tsum.get(m)
        if not t:
            continue
        color = MODEL_COLORS.get(m, "#64748b")
        support = t["tool_support"]
        header_color = "#22c55e" if support else "#ef4444"

        case_rows = ""
        for r in t.get("cases", []):
            calls = r.get("tool_calls_made") or []
            call_list = ""
            for c in calls:
                args_str = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
                call_list += f'<div style="margin:.25rem 0"><code>{c["tool"]}({args_str})</code><br><span style="color:#64748b;font-size:.75rem">→ {c["result_preview"][:120]}…</span></div>'
            if not call_list:
                call_list = '<span style="color:#94a3b8">未呼叫任何工具</span>'
            output_preview = (r.get("output") or "")[:400].replace("<", "&lt;")
            case_rows += f"""
            <tr>
              <td style="width:180px"><strong>{r['test_label']}</strong><br>
                <span style="color:#64748b;font-size:.75rem">{r.get('total_time','—')} s | {len(calls)} 次呼叫</span>
              </td>
              <td>{call_list}</td>
              <td><pre style="max-height:160px;overflow-y:auto;font-size:.75rem;white-space:pre-wrap">{output_preview}</pre></td>
            </tr>"""

        cards += f"""
        <div class="card">
          <div class="card-header" style="background:{color}">
            {m} &nbsp;
            <span style="background:{header_color};border-radius:999px;padding:.1rem .5rem;font-size:.75rem">
              {"Tool Use ✓" if support else "Tool Use ✗"} — 共 {t['calls_total']} 次呼叫
            </span>
          </div>
          <table style="margin:0;box-shadow:none;border-radius:0">
            <thead>
              <tr style="background:#f8fafc">
                <th style="color:#475569;background:#f8fafc">測試案例</th>
                <th style="color:#475569;background:#f8fafc">工具呼叫與結果</th>
                <th style="color:#475569;background:#f8fafc">最終回答</th>
              </tr>
            </thead>
            <tbody>{case_rows}</tbody>
          </table>
        </div>"""

    return f'<h2 class="section">Tool Use 詳細結果</h2>{cards}'


def section_bench_detail(bench: list[dict]) -> str:
    """Per-test-case output comparison cards."""
    tests  = list(dict.fromkeys(r["test_id"] for r in bench))
    models = list(dict.fromkeys(r["model"] for r in bench))
    data   = {(r["model"], r["test_id"]): r for r in bench}

    cards = ""
    for tid in tests:
        sample = next((r for r in bench if r["test_id"] == tid), {})
        label  = sample.get("test_label", tid)
        icon   = "🖼" if sample.get("category") == "image" else "📝"
        cards += f'<h3 style="margin:2rem 0 .75rem;color:#475569">{icon} {label}</h3>'
        for m in models:
            r = data.get((m, tid))
            if not r:
                continue
            color = MODEL_COLORS.get(m, "#64748b")
            if r["error"]:
                body = f'<div style="padding:.75rem 1rem;color:#ef4444">⚠ {r["error"]}</div>'
            else:
                vision_tag = ' <span style="background:#8b5cf6;color:#fff;border-radius:999px;padding:.1rem .4rem;font-size:.72rem">含圖</span>' if r.get("vision_used") else ""
                stats = (
                    f'首token: <strong>{r["time_to_first_token"]}s</strong> | '
                    f'總耗時: <strong>{r["total_time"]}s</strong> | '
                    f'速度: <strong>{r["tokens_per_sec"] or "—"} tok/s</strong>'
                    f'{vision_tag}'
                )
                out_text = (r.get("output") or "")[:600].replace("<", "&lt;")
                body = f"""
                <div style="padding:.5rem 1rem;font-size:.8rem;color:#475569;border-bottom:1px solid #e2e8f0">{stats}</div>
                <div style="padding:.75rem 1rem">
                  <pre style="font-size:.76rem;white-space:pre-wrap;max-height:220px;overflow-y:auto">{out_text}</pre>
                </div>"""
            cards += f"""
            <div class="card" style="margin-bottom:.5rem">
              <div class="card-header" style="background:{color};padding:.4rem 1rem;font-size:.85rem">{m}</div>
              {body}
            </div>"""

    return f'<h2 class="section">Benchmark 詳細輸出</h2>{cards}'


# ── Main report ────────────────────────────────────────────────────────────────

def generate_combined_report(bench: list[dict], tool: list[dict], out_path: Path):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bsum = bench_summary(bench)
    tsum = tool_summary(tool)

    overview   = section_overview(bsum, tsum)
    matrix     = section_capability_matrix(bsum, tsum)
    tool_detail = section_tool_detail(tsum)
    bench_detail = section_bench_detail(bench)

    # Recommendation
    tool_capable = [m for m in MODEL_ORDER if tsum.get(m, {}).get("tool_support")]
    rec_tool = tool_capable[0] if tool_capable else "（無）"
    _models_in_bsum = [m for m in MODEL_ORDER if m in bsum]
    fastest = min(_models_in_bsum, key=lambda m: bsum[m]["text_avg_time"]) if _models_in_bsum else "（無）"
    best_qual = "gemma4:latest" if "gemma4:latest" in bsum else fastest

    rec_html = f"""
    <h2 class="section">建議</h2>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem">
      <div style="background:#fff;border-radius:10px;padding:1.25rem;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:4px solid #22c55e">
        <div style="font-size:.75rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:.5rem">LINE Bot 生產環境</div>
        <div style="font-size:1.1rem;font-weight:700;color:#22c55e">{fastest}</div>
        <div style="font-size:.8rem;color:#64748b;margin-top:.4rem">速度最快，適合即時回覆</div>
      </div>
      <div style="background:#fff;border-radius:10px;padding:1.25rem;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:4px solid #6366f1">
        <div style="font-size:.75rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:.5rem">需查詢法規資料庫</div>
        <div style="font-size:1.1rem;font-weight:700;color:#6366f1">{rec_tool}</div>
        <div style="font-size:.8rem;color:#64748b;margin-top:.4rem">支援 Tool Use，能主動查 DB</div>
      </div>
      <div style="background:#fff;border-radius:10px;padding:1.25rem;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:4px solid #0ea5e9">
        <div style="font-size:.75rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:.5rem">離線批次稽核報告</div>
        <div style="font-size:1.1rem;font-weight:700;color:#0ea5e9">{best_qual}</div>
        <div style="font-size:.8rem;color:#64748b;margin-top:.4rem">輸出品質最高，不在乎速度</div>
      </div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM 完整評測報告 — {run_time}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",system-ui,sans-serif;background:#f8fafc;color:#1e293b;padding:2rem}}
  h1{{font-size:1.7rem;margin-bottom:.3rem}}
  .meta{{color:#64748b;font-size:.85rem;margin-bottom:2rem}}
  h2.section{{font-size:1.05rem;margin:2.5rem 0 .75rem;color:#334155;
    border-left:4px solid #6366f1;padding-left:.6rem}}
  h3{{font-size:.95rem;color:#475569}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
    overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:1.5rem}}
  th{{background:#6366f1;color:#fff;padding:.6rem 1rem;text-align:left;font-size:.8rem}}
  td{{padding:.55rem 1rem;border-bottom:1px solid #e2e8f0;font-size:.82rem;vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  .card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);
    margin-bottom:1rem;overflow:hidden}}
  .card-header{{color:#fff;padding:.5rem 1rem;font-weight:600;font-size:.88rem}}
  code{{background:#f1f5f9;padding:.1rem .35rem;border-radius:4px;font-size:.77rem}}
  pre{{font-family:"Cascadia Code","Consolas",monospace;font-size:.78rem;
    white-space:pre-wrap;word-break:break-word;color:#334155}}
</style>
</head>
<body>
<h1>LLM 完整評測報告</h1>
<div class="meta">
  產生時間：{run_time} ｜
  測試模型：{len(bsum)} 個 ｜
  Benchmark 案例：{len(set(r['test_id'] for r in bench))} 項 ｜
  Tool Use 案例：{len(set(r['test_id'] for r in tool))} 項
</div>

{rec_html}
{overview}
{matrix}
{tool_detail}
{bench_detail}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"報告已儲存 → {out_path.resolve()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="outputs/benchmark_*.json")
    parser.add_argument("--tool",  default="outputs/toolbench_*.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bench = load_latest(args.bench)
    tool  = load_latest(args.tool)

    if not bench:
        print("找不到 benchmark JSON，請先執行 python benchmark.py")
        return
    if not tool:
        print("找不到 toolbench JSON，請先執行 python benchmark.py --tools")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else BASE_DIR / "outputs" / f"full_report_{ts}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate_combined_report(bench, tool, out_path)


if __name__ == "__main__":
    main()
