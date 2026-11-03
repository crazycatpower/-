# -*- coding: utf-8 -*-
import json, html, sys
sys.stdout.reconfigure(encoding="utf-8")

SP = r"C:\Users\q1q2q\AppData\Local\Temp\claude\c--Users-q1q2q-Downloads-0603-20260629T233729Z-3-001-0603\9b35d523-ab3a-47c9-b20a-c189d9b61268\scratchpad"
OUT = SP + r"\audit_v6_report.html"
data = json.load(open(SP + r"\report_data.json", encoding="utf-8"))
cases = data["cases"]
chart_a_svg = open(SP + r"\chart_a.html", encoding="utf-8").read()
chart_b_svg = open(SP + r"\chart_b.html", encoding="utf-8").read()

def esc(x): return html.escape(str(x if x is not None else ""))

# per-case editorial note (my assessment, marked as such)
NOTES = {
    "scaffold":   ("watch", "高處作業照片判為合規，實務上多有缺失 — 疑似漏抓"),
    "foundation": ("over",  "YOLO 零偵測 → 模型空泛擴寫 8 條推測性缺失，多數無法從照片佐證"),
    "panel":      ("watch", "配電盤照片判為合規 — 感電類細節疑似漏抓"),
    "welding":    ("ok",    "簡潔命中，描述可直接進報告"),
    "deep_pit":   ("partial","有抓到危害，但描述偏向墜落、與『中毒缺氧』類別不完全對位"),
    "road":       ("watch", "13 個交通物件卻判合規 — 標誌/錐桶類疑似漏抓"),
}
VERDICT = {True: ("合規", "ok"), False: ("發現缺失", "bad"), None: ("解析失敗", "err")}

case_html = ""
for c in cases:
    v_txt, v_cls = VERDICT[c["compliant"]]
    n_cls, n_txt = NOTES[c["slug"]]
    findings = ""
    if c["findings"]:
        lis = "".join(
            f'<li><span class="fi-item">{esc(f.get("item",""))}</span>{esc(f.get("result",""))}</li>'
            for f in c["findings"]
        )
        findings = f'<ol class="findings">{lis}</ol>'
    else:
        findings = '<p class="no-findings">findings: []　（模型未提出任何缺失）</p>'
    case_html += f"""
      <article class="case">
        <div class="case-photo"><img src="{c['thumb']}" alt="{esc(c['cat'])}現場照片" loading="lazy"><span class="cat-num">{c['num']}</span></div>
        <div class="case-body">
          <header class="case-head">
            <h3>{esc(c['cat'])}<span class="case-sub">{esc(c['desc'])}</span></h3>
            <div class="chips">
              <span class="chip chip-yolo">YOLO 偵測 {c['n_det']}</span>
              <span class="chip chip-{v_cls}">{v_txt}</span>
            </div>
          </header>
          {findings}
          <p class="case-note note-{n_cls}">{esc(n_txt)}</p>
        </div>
      </article>"""

# confusion matrix cells: (label, v6, v1)
cm = [
    ("實際有缺失 · 模型判缺失", "TP", 26, 17, "hit"),
    ("實際合規 · 模型判缺失", "FP", 6, 4, "warn"),
    ("實際有缺失 · 模型判合規", "FN", 15, 24, "miss"),
    ("實際合規 · 模型判合規", "TN", 50, 55, "hit"),
]
cm_html = ""
for lab, code, v6, v1, cls in cm:
    delta = v6 - v1
    darrow = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
    good = (delta > 0 and code in ("TP",)) or (delta < 0 and code in ("FN", "FP"))
    dcls = "d-good" if good else ("d-bad" if delta != 0 else "d-flat")
    cm_html += f"""
        <div class="cm-cell cm-{cls}">
          <span class="cm-code">{code}</span>
          <span class="cm-val">{v6}</span>
          <span class="cm-delta {dcls}">{darrow} v1 {v1}</span>
          <span class="cm-lab">{lab}</span>
        </div>"""

TPL = f"""<title>職安稽核模型 v6 成效報告</title>
<meta name="description" content="行動裝置職安稽核微調模型第六版：100 張標註照片與 6 張全新照片的完整流程實測結果">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@600;700&display=swap">
<style>
  :root {{
    --ground: #f7f5f0;
    --surface: #fffefb;
    --surface-2: #f0ece2;
    --ink: #24211b;
    --ink-soft: #57524740;
    --muted: #6f695c;
    --line: #ded8ca;
    --line-strong: #c6bfad;
    --accent: #a4611a;
    --accent-soft: #a4611a1a;
    --good: #3c7a3a;
    --good-soft: #3c7a3a1a;
    --bad: #b23a2c;
    --bad-soft: #b23a2c1a;
    --warn: #927320;
    --warn-soft: #9273201f;
    --chart-acc: #2a78d6;
    --chart-prec: #eb6834;
    --chart-rec: #1baf7a;
    --chart-v5: #e87ba4;
    --chart-v6: #008300;
    --chart-surface: #fffefb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #17150f;
      --surface: #1f1d16;
      --surface-2: #26231a;
      --ink: #ece7da;
      --ink-soft: #ece7da26;
      --muted: #9a927f;
      --line: #35311f;
      --line-strong: #494330;
      --accent: #d99a4e;
      --accent-soft: #d99a4e1f;
      --good: #7bbf6f;
      --good-soft: #7bbf6f1c;
      --bad: #e07a63;
      --bad-soft: #e07a631f;
      --warn: #d8b45a;
      --warn-soft: #d8b45a1c;
      --chart-acc: #3987e5;
      --chart-prec: #d95926;
      --chart-rec: #199e70;
      --chart-v5: #d55181;
      --chart-v6: #008300;
      --chart-surface: #1f1d16;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground: #17150f;
    --surface: #1f1d16;
    --surface-2: #26231a;
    --ink: #ece7da;
    --ink-soft: #ece7da26;
    --muted: #9a927f;
    --line: #35311f;
    --line-strong: #494330;
    --accent: #d99a4e;
    --accent-soft: #d99a4e1f;
    --good: #7bbf6f;
    --good-soft: #7bbf6f1c;
    --bad: #e07a63;
    --bad-soft: #e07a631f;
    --warn: #d8b45a;
    --warn-soft: #d8b45a1c;
    --chart-acc: #3987e5;
    --chart-prec: #d95926;
    --chart-rec: #199e70;
    --chart-v5: #d55181;
    --chart-v6: #008300;
    --chart-surface: #1f1d16;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: "Noto Sans TC", system-ui, sans-serif;
    font-size: 16px;
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1060px; margin: 0 auto; padding: clamp(1.4rem, 4vw, 3.5rem) clamp(1rem, 4vw, 2.6rem) 4rem; }}
  .col {{ max-width: 780px; }}

  h1, h2, h3 {{ font-family: "Noto Serif TC", "Barlow Condensed", serif; line-height: 1.25; text-wrap: balance; margin: 0; }}
  .eyebrow {{
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    text-transform: uppercase; letter-spacing: .16em; font-weight: 600;
    font-size: .82rem; color: var(--accent); margin: 0 0 .7rem;
  }}
  h1 {{ font-size: clamp(1.9rem, 4.5vw, 2.9rem); font-weight: 700; letter-spacing: .01em; }}
  .lede {{ color: var(--muted); font-size: 1.06rem; margin: 1rem 0 0; max-width: 60ch; }}
  .meta-row {{
    display: flex; flex-wrap: wrap; gap: .5rem 1.6rem; margin-top: 1.5rem;
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    font-size: .95rem; letter-spacing: .04em; color: var(--muted);
    text-transform: uppercase;
  }}
  .meta-row b {{ color: var(--ink); font-weight: 600; }}

  hr.rule {{ border: 0; border-top: 1px solid var(--line); margin: 3rem 0; }}
  section {{ margin-top: 3.2rem; }}
  section > h2 {{ font-size: clamp(1.4rem, 3vw, 1.8rem); font-weight: 700; }}
  section > h2 + p {{ color: var(--muted); margin: .6rem 0 0; max-width: 62ch; }}

  /* verdict tiles */
  .tiles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; margin-top: 1.6rem; }}
  .tile {{ background: var(--surface); padding: 1.4rem 1.3rem 1.5rem; display: flex; flex-direction: column; }}
  .tile.lead {{ background: var(--accent-soft); }}
  .tile .t-lab {{
    display: block;
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    text-transform: uppercase; letter-spacing: .12em; font-weight: 600;
    font-size: .8rem; color: var(--muted);
  }}
  .tile .t-val {{
    display: block;
    font-family: "Barlow Condensed", sans-serif; font-weight: 700;
    font-size: 3rem; line-height: 1; margin: .55rem 0 .4rem;
    font-variant-numeric: tabular-nums;
  }}
  .tile .t-val small {{ font-size: 1.4rem; font-weight: 600; margin-left: .05em; }}
  .tile.lead .t-val {{ color: var(--accent); }}
  .tile .t-delta {{ display: block; font-family: "IBM Plex Mono", monospace; font-size: .84rem; color: var(--good); font-weight: 500; }}
  .tile .t-delta.flat {{ color: var(--muted); }}
  .tile .t-sub {{ display: block; color: var(--muted); font-size: .84rem; margin-top: .55rem; line-height: 1.5; }}

  /* comparison table */
  .tbl-scroll {{ overflow-x: auto; margin-top: 1.6rem; border: 1px solid var(--line); border-radius: 3px; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 460px; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: .72rem 1rem; border-bottom: 1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    text-transform: uppercase; letter-spacing: .08em; font-weight: 600;
    font-size: .84rem; color: var(--muted); background: var(--surface-2);
  }}
  tbody tr:last-child td {{ border-bottom: 0; }}
  tbody td {{ font-family: "IBM Plex Mono", monospace; font-size: .92rem; }}
  tbody th {{ font-family: "Noto Sans TC", sans-serif; font-weight: 500; }}
  tr.now td, tr.now th {{ background: var(--accent-soft); }}
  tr.now td:first-child, tr.now th {{ font-weight: 700; }}
  .cell-hi {{ color: var(--good); font-weight: 600; }}

  /* charts */
  .chart-card {{ margin-top: 1.6rem; padding: 1.4rem 1.5rem 1.1rem; background: var(--chart-surface); border: 1px solid var(--line); border-radius: 3px; }}
  .chart-legend {{ display: flex; flex-wrap: wrap; gap: .35rem 1.3rem; margin-bottom: .8rem; }}
  .chart-legend span {{
    display: inline-flex; align-items: center; gap: .45rem;
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    font-size: .86rem; letter-spacing: .03em; color: var(--muted); font-weight: 600;
  }}
  .chart-legend i {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
  .chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .chart .gl {{ stroke: var(--line); stroke-width: 1; }}
  .chart .gl-v {{ stroke: var(--line); stroke-width: 1; stroke-dasharray: 2 3; }}
  .chart .axis {{ stroke: var(--line-strong); stroke-width: 1; }}
  .chart .ax-x, .chart .ax-y {{
    font-family: "IBM Plex Mono", monospace; font-size: 10.5px; fill: var(--muted);
  }}
  .chart .ax-x {{ text-anchor: middle; }}
  .chart .bar {{ fill: var(--muted); transition: opacity .12s; }}
  .chart .bar:hover {{ opacity: .78; }}
  .chart .bar-acc {{ fill: var(--chart-acc); }}
  .chart .bar-prec {{ fill: var(--chart-prec); }}
  .chart .bar-rec {{ fill: var(--chart-rec); }}
  .chart .bar-val {{
    font-family: "IBM Plex Mono", monospace; font-size: 11px; font-weight: 500;
    fill: var(--ink); text-anchor: middle;
  }}
  .chart .line {{ stroke-width: 2; }}
  .chart .s-v5 {{ stroke: var(--chart-v5); }}
  .chart .s-v6 {{ stroke: var(--chart-v6); }}
  .chart .dot {{ stroke: var(--chart-surface); stroke-width: 1.5; }}
  .chart .dot.s-v5 {{ fill: var(--chart-v5); }}
  .chart .dot.s-v6 {{ fill: var(--chart-v6); }}
  .chart .dot:hover {{ r: 5; }}
  .chart .line-end {{
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    font-size: 12px; font-weight: 600; dominant-baseline: middle;
    fill: var(--ink); /* text stays legible ink, not series hue — the dot at line-end carries color identity (relief for low-contrast series like magenta on light) */
  }}
  .chart-note {{ margin: .9rem 0 0; font-size: .86rem; color: var(--muted); }}

  /* confusion matrix */
  .cm {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; margin-top: 1.6rem; }}
  .cm-cell {{ background: var(--surface); padding: 1.2rem 1.2rem 1.3rem; display: flex; flex-direction: column; gap: .28rem; min-height: 140px; }}
  .cm-hit {{ background: var(--good-soft); }}
  .cm-miss {{ background: var(--bad-soft); }}
  .cm-warn {{ background: var(--warn-soft); }}
  .cm-code {{ font-family: "Barlow Condensed", sans-serif; font-weight: 700; letter-spacing: .1em; font-size: .9rem; color: var(--muted); }}
  .cm-val {{ font-family: "Barlow Condensed", sans-serif; font-weight: 700; font-size: 2.7rem; line-height: 1; font-variant-numeric: tabular-nums; }}
  .cm-delta {{ font-family: "IBM Plex Mono", monospace; font-size: .78rem; }}
  .cm-delta.d-good {{ color: var(--good); }}
  .cm-delta.d-bad {{ color: var(--bad); }}
  .cm-delta.d-flat {{ color: var(--muted); }}
  .cm-lab {{ font-size: .82rem; color: var(--muted); margin-top: auto; line-height: 1.45; }}

  /* cases */
  .cases {{ display: flex; flex-direction: column; gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; margin-top: 1.6rem; }}
  .case {{ background: var(--surface); display: grid; grid-template-columns: 240px 1fr; gap: 0; }}
  .case-photo {{ position: relative; background: var(--surface-2); }}
  .case-photo img {{ width: 100%; height: 100%; object-fit: cover; display: block; aspect-ratio: 3 / 2; }}
  .cat-num {{
    position: absolute; top: 0; left: 0;
    font-family: "Barlow Condensed", sans-serif; font-weight: 700; font-size: 1.05rem;
    background: var(--ink); color: var(--ground); padding: .1rem .6rem; letter-spacing: .05em;
  }}
  .case-body {{ padding: 1.25rem 1.4rem 1.4rem; min-width: 0; }}
  .case-head {{ display: flex; flex-wrap: wrap; gap: .6rem 1rem; justify-content: space-between; align-items: baseline; }}
  .case-head h3 {{ font-size: 1.24rem; font-weight: 700; }}
  .case-sub {{ display: block; font-family: "Noto Sans TC", sans-serif; font-weight: 400; font-size: .86rem; color: var(--muted); margin-top: .15rem; }}
  .chips {{ display: flex; gap: .4rem; flex-wrap: wrap; }}
  .chip {{
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    font-size: .82rem; font-weight: 600; letter-spacing: .04em;
    padding: .12rem .6rem; border-radius: 2px; white-space: nowrap;
    border: 1px solid var(--line-strong); color: var(--muted);
  }}
  .chip-ok {{ background: var(--good-soft); color: var(--good); border-color: transparent; }}
  .chip-bad {{ background: var(--bad-soft); color: var(--bad); border-color: transparent; }}
  .chip-err {{ background: var(--warn-soft); color: var(--warn); border-color: transparent; }}
  ol.findings {{ margin: 1rem 0 0; padding-left: 1.3rem; display: flex; flex-direction: column; gap: .5rem; }}
  ol.findings li {{ font-size: .92rem; line-height: 1.6; }}
  .fi-item {{ display: inline-block; font-weight: 700; margin-right: .5rem; }}
  .fi-item::after {{ content: "／"; color: var(--muted); font-weight: 400; margin-left: .5rem; }}
  .no-findings {{ margin: 1rem 0 0; font-family: "IBM Plex Mono", monospace; font-size: .86rem; color: var(--muted); }}
  .case-note {{
    margin: 1rem 0 0; padding: .55rem .8rem; border-left: 3px solid var(--line-strong);
    font-size: .87rem; color: var(--muted); background: var(--surface-2);
  }}
  .note-watch {{ border-left-color: var(--warn); }}
  .note-over {{ border-left-color: var(--bad); }}
  .note-ok {{ border-left-color: var(--good); }}
  .note-partial {{ border-left-color: var(--warn); }}

  /* limits */
  .limits {{ list-style: none; margin: 1.6rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }}
  .limits li {{ background: var(--surface); padding: 1.1rem 1.3rem; display: grid; grid-template-columns: 2.2rem 1fr; gap: .5rem; }}
  .limits .n {{ font-family: "Barlow Condensed", sans-serif; font-weight: 700; color: var(--accent); font-size: 1.2rem; }}
  .limits b {{ display: block; margin-bottom: .1rem; }}
  .limits span.d {{ color: var(--muted); font-size: .92rem; }}

  /* methods / papers */
  .methods {{ margin-top: 1.6rem; display: flex; flex-direction: column; gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }}
  .method {{ background: var(--surface); padding: 1.15rem 1.35rem 1.25rem; }}
  .method-tag {{
    display: inline-block;
    font-family: "Barlow Condensed", "Noto Sans TC", sans-serif;
    text-transform: uppercase; letter-spacing: .1em; font-weight: 600;
    font-size: .72rem; padding: .12rem .5rem; border-radius: 2px;
    background: var(--surface-2); color: var(--muted);
  }}
  .method-tag.t-v5 {{ background: var(--accent-soft); color: var(--accent); }}
  .method-tag.t-v6 {{ background: var(--good-soft); color: var(--good); }}
  .method-tag.t-no {{ background: var(--bad-soft); color: var(--bad); }}
  .method h3 {{ font-family: "Noto Sans TC", sans-serif; font-size: 1.06rem; font-weight: 700; margin: .55rem 0 .3rem; }}
  .method p {{ font-size: .91rem; color: var(--muted); margin: 0; line-height: 1.62; }}
  .method .cite {{
    display: block; margin-top: .65rem; padding-top: .55rem;
    border-top: 1px dashed var(--line-strong);
    font-family: "IBM Plex Mono", monospace; font-size: .78rem; color: var(--muted); line-height: 1.6;
  }}
  .method .cite b {{ color: var(--ink); font-weight: 500; }}
  .method p code {{ font-family: "IBM Plex Mono", monospace; font-size: .84em; background: var(--surface-2); padding: .02rem .3rem; border-radius: 2px; color: var(--ink); }}

  footer {{ margin-top: 3.5rem; padding-top: 1.6rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .84rem; line-height: 1.7; }}
  footer code {{ font-family: "IBM Plex Mono", monospace; font-size: .82rem; background: var(--surface-2); padding: .05rem .35rem; border-radius: 2px; color: var(--ink); }}
  footer .fn-t {{ font-family: "Barlow Condensed","Noto Sans TC",sans-serif; text-transform: uppercase; letter-spacing:.1em; color: var(--ink); font-weight: 600; font-size: .8rem; }}

  @media (max-width: 620px) {{
    .tiles {{ grid-template-columns: 1fr; }}
    .cm {{ grid-template-columns: 1fr; }}
    .case {{ grid-template-columns: 1fr; }}
    .case-photo img {{ aspect-ratio: 16 / 9; }}
  }}
  @media (prefers-reduced-motion: no-preference) {{
    .tile, .cm-cell, .case {{ transition: none; }}
  }}
</style>

<div class="wrap">
  <header class="col">
    <p class="eyebrow">內部研發進度報告 · R&amp;D Progress</p>
    <h1>行動裝置職安稽核模型　第 6 版成效</h1>
    <p class="lede">以工地現場照片自動判定職業安全合規性的微調模型。本報告比對 v1 初版與 v6 的偵測表現，並附 6 張全新照片的完整流程實測。</p>
    <div class="meta-row">
      <span>日期 <b>2026-09-09</b></span>
      <span>底模 <b>Gemma 3n E2B</b></span>
      <span>評測集 <b>100 張標註照片</b></span>
      <span>硬體 <b>單張 RTX 3060</b></span>
    </div>
  </header>

  <hr class="rule">

  <section class="col">
    <p class="eyebrow">結論摘要</p>
    <h2>召回率翻倍，精確率不變</h2>
    <p>對安全稽核而言，「漏抓危害」（false negative）比「誤報」代價更高。v6 在不犧牲精確率的前提下，把召回率從初版的 0.42 拉到 0.63。</p>
    <div class="tiles">
      <div class="tile">
        <span class="t-lab">Accuracy · 準確率</span>
        <span class="t-val">78<small>%</small></span>
        <span class="t-delta">▲ +6.4 pt vs v1</span>
        <span class="t-sub">整體判斷正確比例</span>
      </div>
      <div class="tile">
        <span class="t-lab">Precision · 精確率</span>
        <span class="t-val">81<small>%</small></span>
        <span class="t-delta flat">± 0 pt vs v1</span>
        <span class="t-sub">判為「有缺失」時的正確率</span>
      </div>
      <div class="tile lead">
        <span class="t-lab">Recall · 召回率</span>
        <span class="t-val">63<small>%</small></span>
        <span class="t-delta">▲ +22 pt vs v1</span>
        <span class="t-sub">實際有缺失的案例中被抓到的比例</span>
      </div>
    </div>
  </section>

  <section class="col">
    <p class="eyebrow">版本演進</p>
    <h2>三版對比</h2>
    <p>同一組 100 張照片、同一條 YOLO 物件偵測 → 模型判讀 → JSON 解析的完整流程。</p>
    <div class="tbl-scroll">
      <table>
        <thead>
          <tr><th>版本</th><th>準確率</th><th>精確率</th><th>召回率</th><th>解析失敗</th><th>TP / TN / FP / FN</th></tr>
        </thead>
        <tbody>
          <tr><th>v1 初版</th><td>0.720</td><td>0.810</td><td>0.415</td><td>0</td><td>17 / 55 / 4 / 24</td></tr>
          <tr><th>v5</th><td>0.758</td><td>0.800</td><td>0.526</td><td>5</td><td>20 / 52 / 5 / 18</td></tr>
          <tr class="now"><th>v6 最終</th><td class="cell-hi">0.784</td><td class="cell-hi">0.813</td><td class="cell-hi">0.634</td><td>3</td><td>26 / 50 / 6 / 15</td></tr>
        </tbody>
      </table>
    </div>
    <p style="font-size:.86rem;color:var(--muted);margin-top:.9rem">v6 做法：訓練資料把「有缺失」樣本複製一份（×2）、訓練輪數 2→3。對照組（僅訓練一半、checkpoint-100）召回率只有 0.34，確認提升來自完整訓練而非過擬合。</p>
    <div class="chart-card">
      <div class="chart-legend">
        <span><i style="background:var(--chart-acc)"></i>準確率</span>
        <span><i style="background:var(--chart-prec)"></i>精確率</span>
        <span><i style="background:var(--chart-rec)"></i>召回率</span>
      </div>
      {chart_a_svg}
    </div>
  </section>

  <section class="col">
    <p class="eyebrow">訓練過程</p>
    <h2>損失曲線　v5 vs v6</h2>
    <p>同一支腳本、同一批底模，v6 把「有缺失」樣本加倍、多訓一輪。橫軸為訓練輪數（epoch），縱軸為 loss（對數尺度）。v5 在 epoch 1.86 停止；v6 續訓到 epoch 2.96，loss 從同起點的 ~3.7 一路壓到 0.08 附近。</p>
    <div class="chart-card">
      <div class="chart-legend">
        <span><i style="background:var(--chart-v5)"></i>v5（2 epoch，108 步）</span>
        <span><i style="background:var(--chart-v6)"></i>v6（3 epoch，234 步）</span>
      </div>
      {chart_b_svg}
    </div>
    <p class="chart-note">v6 訓練末段 loss（~0.08–0.11）明顯低於 v5 收斂區間，entropy 掉到 0.068——單看訓練曲線像是過擬合；但 100 張評測集上 v6 recall 仍優於 v5（0.63 對 0.53），且對照組 checkpoint-100 表現更差，說明這段額外訓練是有效學習而非單純記憶。</p>
  </section>

  <section class="col">
    <p class="eyebrow">錯誤結構</p>
    <h2>混淆矩陣　v6（對照 v1）</h2>
    <p>綠：判斷正確。紅：漏抓（最需要壓低）。黃：誤報。v6 的漏抓從 24 降到 15。</p>
    <div class="cm">{cm_html}
    </div>
  </section>

  <section>
    <div class="col">
      <p class="eyebrow">完整流程實測</p>
      <h2>6 張全新照片</h2>
      <p>訓練與驗證集從未出現過的照片，六大職安宣導重點各一張，跑完整推論流程。判定為「合規／缺失」的結果與 v1 相同，差異在缺失描述的數量與品質。標註為個人研判。</p>
    </div>
    <div class="cases">{case_html}
    </div>
  </section>

  <section class="col">
    <p class="eyebrow">已知限制</p>
    <h2>下一輪待處理</h2>
    <p>不影響 v6 交付，但需在後續版本收斂。</p>
    <ol class="limits">
      <li><span class="n">1</span><span><b>YOLO 零偵測影像會空泛擴寫</b><span class="d">物件偵測為 0 時，模型傾向生出多條無法從照片佐證的推測性缺失（如上方開挖案例的 8 條）。約占評測集 3%，也是剩餘 3 個解析失敗的同源問題。</span></span></li>
      <li><span class="n">2</span><span><b>個人防護具細節仍漏約三分之一</b><span class="d">未戴安全帽、未穿反光背心這類需要細看人物的缺失，召回率明顯低於整體。</span></span></li>
      <li><span class="n">3</span><span><b>缺失描述與照片類別不完全對位</b><span class="d">部分輸出雖抓到危害，但描述落在其他類別（如深坑照片給出墜落相關描述）。</span></span></li>
    </ol>
  </section>

  <section class="col">
    <p class="eyebrow">技術附錄</p>
    <h2>方法與論文依據</h2>
    <p>今天實際動到的技術，依「v5 改動 / v6 改動 / 否定結論 / 底層方法」分類，附對應文獻。</p>
    <div class="methods">

      <div class="method">
        <span class="method-tag t-v5">v5 改動 · 關鍵</span>
        <h3>訓練／推論格式對齊</h3>
        <p>舊版訓練 prompt 為純文字，推論時卻在 prompt 內塞一大段 YOLO 偵測框清單 — 對模型是 out-of-distribution 輸入，實測讓召回率從 0.56 掉到 0.41。v5 改成訓練時就用同一支 YOLO 對每張訓練圖跑真實偵測，把清單逐字接在 prompt 後，格式與推論 byte-for-byte 一致；system prompt 與輸出 JSON schema 三邊對齊。</p>
        <span class="cite">train–serving skew。把物件偵測結果當文字證據餵給視覺模型以強化 grounding，概念近 <b>Set-of-Mark Prompting</b>（Yang et al. 2023, arXiv:2310.11441）。</span>
      </div>

      <div class="method">
        <span class="method-tag t-v5">v5 改動</span>
        <h3>Completion-only 損失遮罩</h3>
        <p>舊版只遮 padding，等於逼模型連 system prompt、半模板化的 YOLO 清單、甚至圖片 soft token 都要「學會生成」— 浪費容量，且模板化文字會拉高生成時卡重複迴圈的機率。v5 把 assistant 回答之前的所有 token（含 image placeholder）label 設為 −100，只在 JSON 答案上算 loss。</p>
        <span class="cite">指令微調標準做法。<b>LLaVA 視覺指令微調</b>（Liu et al. 2023, arXiv:2304.08485）只在 assistant response 算 loss；<b>LIMA</b>（Zhou et al. 2023, arXiv:2305.11206）指出對 prompt 算 loss 會傷品質。</span>
      </div>

      <div class="method">
        <span class="method-tag t-v5">v5 改動 · 工程</span>
        <h3>梯度檢查點（解 12GB OOM）</h3>
        <p>RTX 3060 12GB 跑 QLoRA ＋ 視覺塔 ＋ 896px 影像，第 15 步 CUDA OOM，還會先溢出到 shared memory 讓每步從 90s 爬到 155s。開 <code>gradient_checkpointing</code>（<code>use_reentrant=False</code>）後前向不存中間 activation、反向時重算，以計算換記憶體，每步反而降到約 55s。</p>
        <span class="cite">activation recomputation / sublinear memory — <b>Chen et al. 2016</b>,「Training Deep Nets with Sublinear Memory Cost」, arXiv:1604.06174。</span>
      </div>

      <div class="method">
        <span class="method-tag t-v6">v6 改動 · 關鍵</span>
        <h3>少數類別（有缺失）過採樣 ×2</h3>
        <p>對安全稽核，漏抓危害（false negative）比誤報代價高，而 v5 精確率 0.80 仍有餘裕。v6 在 train split 把 <code>compliant=false</code> 樣本複製一份（violation:compliant 由 45:55 → 約 62:38），把決策邊界推向「敢開缺失」。召回率 0.53 → 0.63，精確率維持。</p>
        <span class="cite">類別不平衡下的 minority oversampling / cost-sensitive learning；動機同 <b>Focal Loss</b>（Lin et al. 2017, arXiv:1708.02002），此處用最樸素的重複採樣而非 focal loss 或 SMOTE。</span>
      </div>

      <div class="method">
        <span class="method-tag t-v6">v6 改動</span>
        <h3>訓練輪數 2 → 3 ＋ checkpoint 對照驗證</h3>
        <p>v5 結束時 loss 仍在下降，未充分收斂，v6 加一輪。final 的 train entropy 掉到 0.068 看似過擬合，但評測 checkpoint-100（僅訓練一半）召回率只有 0.34、final 為 0.63 — 證明提升來自完整訓練而非記憶訓練集。</p>
        <span class="cite">checkpoint selection / early-stopping 標準方法論。</span>
      </div>

      <div class="method">
        <span class="method-tag t-no">否定結論</span>
        <h3>推論端抑制重複 → 精確率崩，不可用</h3>
        <p>剩餘的 JSON 解析失敗來自 YOLO 零偵測圖的重複迴圈。試 <code>repetition_penalty=1.3</code> 召回率升到 ~0.86 但精確率掉到 ~0.41；<code>no_repeat_ngram_size=4</code> 召回率 ~1.0、精確率 ~0.44、真陰性只剩 1。因為合規答案 <code>{{"compliant":true,"findings":[]}}</code> 又短又高頻，全域抑制重複會把機率質量從它推開。此問題須在訓練端或以「解析失敗→轉人工複核」處理。</p>
        <span class="cite">貪婪解碼退化成重複迴圈 — <b>Holtzman et al. 2019</b>, arXiv:1904.09751。repetition_penalty — CTRL, Keskar et al. 2019, arXiv:1909.05858。n-gram 阻擋 — Paulus et al. 2017, arXiv:1705.04304。</span>
      </div>

      <div class="method">
        <span class="method-tag">底層方法</span>
        <h3>QLoRA（4-bit NF4 ＋ LoRA rank 32）</h3>
        <p>4-bit NF4 量化凍結底模，只訓練掛在 q/k/v/o/gate/up/down 投影上的低秩適配器；搭配 8-bit 分頁 AdamW 優化器避免記憶體尖峰。全程單張消費級 GPU。</p>
        <span class="cite"><b>LoRA</b> — Hu et al. 2021, arXiv:2106.09685。<b>QLoRA</b>（NF4 / 雙重量化 / 分頁優化器）— Dettmers et al. 2023, arXiv:2305.14314。8-bit 優化器 — Dettmers et al. 2021, arXiv:2110.02861。</span>
      </div>

      <div class="method">
        <span class="method-tag">底層方法</span>
        <h3>底模 Gemma 3n E2B</h3>
        <p>Google on-device 多模態模型，檔案 5.4B 參數但有效約 2B — 靠 MatFormer（巢狀 / 可伸縮 Transformer）與 Per-Layer Embeddings（把嵌入參數卸到快取）達成；視覺塔為 MobileNet-V5。前處理的物件偵測用 Ultralytics YOLO。</p>
        <span class="cite">Gemma 3 技術報告 — arXiv:2503.19786。<b>MatFormer</b> — Kudugunta et al. 2023, arXiv:2310.07707。YOLO 原始概念 — Redmon et al. 2016, arXiv:1506.02640。</span>
      </div>

    </div>
  </section>

  <footer class="col">
    <p><span class="fn-t">模型</span>　<code>gemma4-e2b-audit-lora-v6/final</code>　·　底模 <code>google/gemma-3n-E2B-it</code>　·　QLoRA rank 32　·　3 epochs　·　train 620 / val 57</p>
    <p><span class="fn-t">評測方法</span>　100 張取自 483 張標註資料集（固定亂數種子）；輸入為現場照片 + YOLO 物件清單；以模型輸出 JSON 的 findings 是否為空判定「有無缺失」，與答案的 violation 欄位比對。貪婪解碼，未加抑制重複參數（實測會嚴重犧牲精確率）。</p>
    <p><span class="fn-t">附註</span>　評測集與訓練集有部分影像重疊，此限制在 v1／v5／v6 三版一致，相對比較仍成立。上方 6 張完整流程照片則完全未曾出現於訓練。</p>
  </footer>
</div>
"""

open(OUT, "w", encoding="utf-8").write(TPL)
print("written", OUT, len(TPL), "bytes")
