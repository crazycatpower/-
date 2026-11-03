# -*- coding: utf-8 -*-
"""Generate the two SVG chart snippets (grouped bar + loss line) as HTML fragments,
written to chart_a.html / chart_b.html for splicing into build_v6_report.py."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
SP = r"C:\Users\q1q2q\AppData\Local\Temp\claude\c--Users-q1q2q-Downloads-0603-20260629T233729Z-3-001-0603\9b35d523-ab3a-47c9-b20a-c189d9b61268\scratchpad"

# ---------- Chart A: grouped bar, accuracy/precision/recall x v1/v5/v6 ----------
versions = ["v1 初版", "v5", "v6 最終"]
metrics = [("accuracy", "acc", "準確率"), ("precision", "prec", "精確率"), ("recall", "rec", "召回率")]
data = {
    "v1 初版": {"accuracy": .720, "precision": .810, "recall": .415},
    "v5":      {"accuracy": .758, "precision": .800, "recall": .526},
    "v6 最終": {"accuracy": .784, "precision": .813, "recall": .634},
}

W, H = 720, 320
ML, MR, MT, MB = 4, 4, 16, 34
plotW = W - ML - MR
plotH = H - MT - MB
y0 = MT + plotH  # baseline (loss=0 / value=0)

def yv(v):  # value 0..1 -> svg y
    return MT + plotH * (1 - v / 0.9)

n_groups = len(versions)
group_w = plotW / n_groups
bar_gap = 6
group_pad = 26
bar_w = (group_w - group_pad - bar_gap * 2) / 3

svg_a = []
# gridlines + y labels at 0,20,40,60,80%
for pct in (0, 20, 40, 60, 80):
    y = yv(pct / 100)
    svg_a.append(f'<line class="gl" x1="{ML}" x2="{W-MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
    svg_a.append(f'<text class="ax-y" x="0" y="{y-3:.1f}">{pct}%</text>')
# baseline emphasized
svg_a.append(f'<line class="axis" x1="{ML}" x2="{W-MR}" y1="{y0:.1f}" y2="{y0:.1f}"/>')

for gi, ver in enumerate(versions):
    gx = ML + gi * group_w + group_pad / 2
    for mi, (key, short, label) in enumerate(metrics):
        v = data[ver][key]
        bx = gx + mi * (bar_w + bar_gap)
        by = yv(v)
        bh = y0 - by
        cls = f"bar-{short}"
        pct_txt = f"{v*100:.1f}%"
        svg_a.append(
            f'<g class="bar-g">'
            f'<rect class="bar {cls}" x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="2.5">'
            f'<title>{ver} · {label} {pct_txt}</title></rect>'
            f'<text class="bar-val" x="{bx+bar_w/2:.1f}" y="{by-6:.1f}">{v*100:.0f}</text>'
            f'</g>'
        )
    # x label
    cx = gx + (3*bar_w + 2*bar_gap)/2
    svg_a.append(f'<text class="ax-x" x="{cx:.1f}" y="{H-10}">{ver}</text>')

chart_a = f'''<svg class="chart chart-bar" viewBox="0 0 {W} {H}" role="img" aria-label="v1、v5、v6 三版準確率、精確率、召回率長條圖">
  {''.join(svg_a)}
</svg>'''
open(SP + r"\chart_a.html", "w", encoding="utf-8").write(chart_a)

# ---------- Chart B: loss curves, v5 vs v6, x=epoch, y=loss (log scale) ----------
v5_pts = [(0.1878,3.571),(0.3756,1.493),(0.5634,1.592),(0.7512,0.8559),(0.939,1.09),
          (1.113,0.6145),(1.3,0.5218),(1.488,0.6623),(1.676,0.7085),(1.864,0.4773)]
v6_pts = [(0.129,3.775),(0.2581,1.638),(0.3871,1.164),(0.5161,1.416),(0.6452,1.078),
          (0.7742,0.8264),(0.9032,0.6227),(1.026,0.5881),(1.155,0.304),(1.284,0.3985),
          (1.413,0.5219),(1.542,0.2201),(1.671,0.2928),(1.8,0.2428),(1.929,0.1721),
          (2.052,0.2044),(2.181,0.134),(2.31,0.08464),(2.439,0.1061),(2.568,0.07854),
          (2.697,0.1095),(2.826,0.1176),(2.955,0.09123)]

import math
W2, H2 = 720, 340
ML2, MR2, MT2, MB2 = 34, 46, 16, 34
plotW2 = W2 - ML2 - MR2
plotH2 = H2 - MT2 - MB2
X_MAX = 3.0
Y_MIN, Y_MAX = 0.07, 4.2

def xs(e):
    return ML2 + plotW2 * (e / X_MAX)

def ys(loss):
    lo, hi = math.log10(Y_MIN), math.log10(Y_MAX)
    t = (math.log10(loss) - lo) / (hi - lo)
    return MT2 + plotH2 * (1 - t)

svg_b = []
for tick in (4, 2, 1, 0.5, 0.25, 0.1):
    y = ys(tick)
    svg_b.append(f'<line class="gl" x1="{ML2}" x2="{W2-MR2}" y1="{y:.1f}" y2="{y:.1f}"/>')
    lab = f"{tick:g}"
    svg_b.append(f'<text class="ax-y" x="{ML2-8}" y="{y+3:.1f}" text-anchor="end">{lab}</text>')
for e in (0, 1, 2, 3):
    x = xs(e)
    svg_b.append(f'<line class="gl-v" x1="{x:.1f}" x2="{x:.1f}" y1="{MT2}" y2="{MT2+plotH2}"/>')
    svg_b.append(f'<text class="ax-x" x="{x:.1f}" y="{H2-10}">epoch {e}</text>')

def build_series(pts, cls, name):
    path = "M " + " L ".join(f"{xs(e):.1f} {ys(l):.1f}" for e, l in pts)
    out = [f'<path class="line {cls}" d="{path}" fill="none"/>']
    for e, l in pts:
        out.append(f'<circle class="dot {cls}" cx="{xs(e):.1f}" cy="{ys(l):.1f}" r="3.2"><title>{name} epoch {e:.2f} · loss {l:.3f}</title></circle>')
    lx, ly = pts[-1]
    out.append(f'<text class="line-end {cls}" x="{xs(lx)+8:.1f}" y="{ys(ly)+4:.1f}">{name}</text>')
    return "".join(out)

svg_b.append(build_series(v5_pts, "s-v5", "v5（2 epoch）"))
svg_b.append(build_series(v6_pts, "s-v6", "v6（3 epoch）"))

chart_b = f'''<svg class="chart chart-line" viewBox="0 0 {W2} {H2}" role="img" aria-label="v5 與 v6 訓練損失曲線，橫軸為訓練輪數，縱軸為對數尺度的 loss">
  {''.join(svg_b)}
</svg>'''
open(SP + r"\chart_b.html", "w", encoding="utf-8").write(chart_b)

print("chart_a bytes", len(chart_a))
print("chart_b bytes", len(chart_b))
