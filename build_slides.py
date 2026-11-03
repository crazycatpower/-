# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
SP = r"C:\Users\q1q2q\AppData\Local\Temp\claude\c--Users-q1q2q-Downloads-0603-20260629T233729Z-3-001-0603\9b35d523-ab3a-47c9-b20a-c189d9b61268\scratchpad"
main_svg = open(SP + r"\slide_chart_main.html", encoding="utf-8").read()
abl_svg = open(SP + r"\slide_chart_ablation.html", encoding="utf-8").read()
loss_svg = open(SP + r"\slide_chart_loss.html", encoding="utf-8").read()

TPL = f"""<title>職安稽核模型圖表</title>
<meta name="description" content="v1/v5/v6 三版指標長條圖、控制變量對照、v5 vs v6 訓練損失曲線 — 簡報用單頁圖表">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@600;700&display=swap">
<style>
  :root {{
    --ground: #f7f5f0; --surface: #fffefb; --surface-2: #f0ece2;
    --ink: #24211b; --muted: #6f695c; --line: #ded8ca; --line-strong: #c6bfad;
    --accent: #a4611a; --accent-soft: #a4611a1a;
    --chart-acc: #2a78d6; --chart-prec: #eb6834; --chart-rec: #1baf7a;
    --chart-v5: #e87ba4; --chart-v6: #008300; --chart-surface: #fffefb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #17150f; --surface: #1f1d16; --surface-2: #26231a;
      --ink: #ece7da; --muted: #9a927f; --line: #35311f; --line-strong: #494330;
      --accent: #d99a4e; --accent-soft: #d99a4e1f;
      --chart-acc: #3987e5; --chart-prec: #d95926; --chart-rec: #199e70;
      --chart-v5: #d55181; --chart-v6: #008300; --chart-surface: #1f1d16;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground: #17150f; --surface: #1f1d16; --surface-2: #26231a;
    --ink: #ece7da; --muted: #9a927f; --line: #35311f; --line-strong: #494330;
    --accent: #d99a4e; --accent-soft: #d99a4e1f;
    --chart-acc: #3987e5; --chart-prec: #d95926; --chart-rec: #199e70;
    --chart-v5: #d55181; --chart-v6: #008300; --chart-surface: #1f1d16;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--ground); color: var(--ink); font-family: "Noto Sans TC", system-ui, sans-serif; line-height: 1.6; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: clamp(1.2rem,4vw,2.6rem) clamp(1rem,4vw,2rem) 3rem; }}
  h1 {{ font-family: "Noto Serif TC", serif; font-size: clamp(1.5rem,3.4vw,2.1rem); font-weight: 700; margin: 0 0 .3rem; text-wrap: balance; }}
  .sub {{ color: var(--muted); font-size: .95rem; margin: 0 0 2rem; }}
  .slide {{ margin-top: 2.6rem; }}
  .slide h2 {{ font-family: "Noto Serif TC", serif; font-size: 1.3rem; font-weight: 700; margin: 0 0 .3rem; }}
  .slide .cap {{ color: var(--muted); font-size: .9rem; margin: 0 0 1rem; max-width: 68ch; }}
  .card {{ background: var(--chart-surface); border: 1px solid var(--line); border-radius: 4px; padding: 1.3rem 1.5rem 1.1rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: .3rem 1.2rem; margin-bottom: .8rem; }}
  .legend span {{ display: inline-flex; align-items: center; gap: .4rem; font-family: "Barlow Condensed","Noto Sans TC",sans-serif; font-size: .88rem; font-weight: 600; color: var(--muted); letter-spacing: .02em; }}
  .legend i {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
  .chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
  .chart .gl {{ stroke: var(--line); stroke-width: 1; }}
  .chart .gl-v {{ stroke: var(--line); stroke-width: 1; stroke-dasharray: 2 3; }}
  .chart .axis {{ stroke: var(--line-strong); stroke-width: 1; }}
  .chart .ax-x, .chart .ax-y {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; fill: var(--muted); }}
  .chart .ax-x {{ text-anchor: middle; }}
  .chart .ax-x-emph {{ fill: var(--ink); font-weight: 600; }}
  .chart .bar {{ fill: var(--muted); }}
  .chart .bar-acc {{ fill: var(--chart-acc); }}
  .chart .bar-prec {{ fill: var(--chart-prec); }}
  .chart .bar-rec {{ fill: var(--chart-rec); }}
  .chart .bar-dim {{ opacity: .45; }}
  .chart .bar-val {{ font-family: "IBM Plex Mono", monospace; font-size: 12.5px; font-weight: 600; fill: var(--ink); text-anchor: middle; }}
  .chart .bar-val-dim {{ fill: var(--muted); font-weight: 500; }}
  .chart .line {{ stroke-width: 2.2; }}
  .chart .s-v5 {{ stroke: var(--chart-v5); }}
  .chart .s-v6 {{ stroke: var(--chart-v6); }}
  .chart .dot {{ stroke: var(--chart-surface); stroke-width: 1.5; }}
  .chart .dot.s-v5 {{ fill: var(--chart-v5); }}
  .chart .dot.s-v6 {{ fill: var(--chart-v6); }}
  .chart .line-ext {{ stroke-width: 1.6; opacity: .7; }}
  .chart .line-ext.s-v5 {{ stroke: var(--chart-v5); }}
  .chart .line-ext.s-v6 {{ stroke: var(--chart-v6); }}
  .chart .dot-end {{ fill: var(--chart-surface); stroke-width: 2; }}
  .chart .dot-end.s-v5 {{ stroke: var(--chart-v5); }}
  .chart .dot-end.s-v6 {{ stroke: var(--chart-v6); }}
  .chart .line-end {{ font-family: "Barlow Condensed","Noto Sans TC",sans-serif; font-size: 13px; font-weight: 700; fill: var(--ink); dominant-baseline: middle; }}
  @media (max-width: 560px) {{ .card {{ padding: 1rem 1rem .9rem; }} }}
</style>

<div class="wrap">
  <h1>職安稽核模型　圖表</h1>
  <p class="sub">2026-09 · gemma-3n-E2B QLoRA 微調 · 100 張標註照片、YOLO 真實流程評測</p>

  <div class="slide">
    <h2>三版核心指標</h2>
    <p class="cap">v6 召回率較初版 +22 個百分點，精確率沒有犧牲。</p>
    <div class="card">
      <div class="legend">
        <span><i style="background:var(--chart-acc)"></i>準確率</span>
        <span><i style="background:var(--chart-prec)"></i>精確率</span>
        <span><i style="background:var(--chart-rec)"></i>召回率</span>
      </div>
      {main_svg}
    </div>
  </div>

  <div class="slide">
    <h2>控制變量對照</h2>
    <p class="cap">以 v5 為基準，各改一個變量：拿掉 YOLO 清單、或單獨多訓一輪——兩者都比 v5 差，證實 v6 的進步來自過採樣本身。</p>
    <div class="card">
      <div class="legend">
        <span><i style="background:var(--chart-acc)"></i>準確率</span>
        <span><i style="background:var(--chart-prec)"></i>精確率</span>
        <span><i style="background:var(--chart-rec)"></i>召回率</span>
      </div>
      {abl_svg}
    </div>
  </div>

  <div class="slide">
    <h2>訓練損失曲線　v5 vs v6</h2>
    <p class="cap">橫軸訓練輪數，縱軸 loss（對數尺度）。實線是每 10 步記一次的 loss；虛線＋空心終點是訓練實際跑完的位置（v5 滿 108 步／2 epoch，v6 滿 234 步／3 epoch）——log 頻率沒抓到最後一步，不代表提前中斷。</p>
    <div class="card">
      <div class="legend">
        <span><i style="background:var(--chart-v5)"></i>v5（2 epoch）</span>
        <span><i style="background:var(--chart-v6)"></i>v6（3 epoch）</span>
      </div>
      {loss_svg}
    </div>
  </div>
</div>
"""
open(SP + r"\audit_charts_slides.html", "w", encoding="utf-8").write(TPL)
print("written", len(TPL))
