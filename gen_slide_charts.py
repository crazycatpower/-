# -*- coding: utf-8 -*-
import math, sys
sys.stdout.reconfigure(encoding="utf-8")
SP = r"C:\Users\q1q2q\AppData\Local\Temp\claude\c--Users-q1q2q-Downloads-0603-20260629T233729Z-3-001-0603\9b35d523-ab3a-47c9-b20a-c189d9b61268\scratchpad"

METRICS = [("accuracy", "acc", "準確率"), ("precision", "prec", "精確率"), ("recall", "rec", "召回率")]

def grouped_bar(groups, W=880, H=380, y_max=0.9):
    ML, MR, MT, MB = 4, 4, 20, 40
    plotW, plotH = W - ML - MR, H - MT - MB
    y0 = MT + plotH
    def yv(v): return MT + plotH * (1 - v / y_max)
    n = len(groups)
    group_w = plotW / n
    bar_gap, group_pad = 7, 22
    bar_w = (group_w - group_pad - bar_gap * 2) / 3
    out = []
    for pct in (0, 20, 40, 60, 80):
        y = yv(pct/100)
        out.append(f'<line class="gl" x1="{ML}" x2="{W-MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax-y" x="0" y="{y-4:.1f}">{pct}%</text>')
    out.append(f'<line class="axis" x1="{ML}" x2="{W-MR}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
    for gi, (label, vals, emph) in enumerate(groups):
        gx = ML + gi * group_w + group_pad / 2
        for mi, (key, short, mlabel) in enumerate(METRICS):
            v = vals[key]
            bx = gx + mi * (bar_w + bar_gap)
            by = yv(v)
            bh = y0 - by
            out.append(
                f'<g class="bar-g"><rect class="bar bar-{short}{" bar-dim" if not emph else ""}" '
                f'x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3">'
                f'<title>{label} · {mlabel} {v*100:.1f}%</title></rect>'
                f'<text class="bar-val{" bar-val-dim" if not emph else ""}" x="{bx+bar_w/2:.1f}" y="{by-7:.1f}">{v*100:.0f}</text></g>'
            )
        cx = gx + (3*bar_w + 2*bar_gap)/2
        cls = "ax-x ax-x-emph" if emph else "ax-x"
        out.append(f'<text class="{cls}" x="{cx:.1f}" y="{H-14}">{label}</text>')
    return f'<svg class="chart chart-bar" viewBox="0 0 {W} {H}" role="img">{"".join(out)}</svg>'

chart_main = grouped_bar([
    ("v1 初版", {"accuracy": .720, "precision": .810, "recall": .415}, False),
    ("v5",      {"accuracy": .758, "precision": .800, "recall": .526}, False),
    ("v6 最終", {"accuracy": .784, "precision": .813, "recall": .634}, True),
])

chart_ablation = grouped_bar([
    ("v5（基準）",         {"accuracy": .758, "precision": .800, "recall": .526}, True),
    ("v5-noyolo（無YOLO）", {"accuracy": .585, "precision": .490, "recall": .658}, False),
    ("v5-e3（多訓一輪）",   {"accuracy": .684, "precision": .645, "recall": .500}, False),
])

open(SP + r"\slide_chart_main.html", "w", encoding="utf-8").write(chart_main)
open(SP + r"\slide_chart_ablation.html", "w", encoding="utf-8").write(chart_ablation)

# ---------- loss curve (reuse v5/v6) ----------
v5_pts = [(0.1878,3.571),(0.3756,1.493),(0.5634,1.592),(0.7512,0.8559),(0.939,1.09),
          (1.113,0.6145),(1.3,0.5218),(1.488,0.6623),(1.676,0.7085),(1.864,0.4773)]
v6_pts = [(0.129,3.775),(0.2581,1.638),(0.3871,1.164),(0.5161,1.416),(0.6452,1.078),
          (0.7742,0.8264),(0.9032,0.6227),(1.026,0.5881),(1.155,0.304),(1.284,0.3985),
          (1.413,0.5219),(1.542,0.2201),(1.671,0.2928),(1.8,0.2428),(1.929,0.1721),
          (2.052,0.2044),(2.181,0.134),(2.31,0.08464),(2.439,0.1061),(2.568,0.07854),
          (2.697,0.1095),(2.826,0.1176),(2.955,0.09123)]
W2, H2 = 880, 380
ML2, MR2, MT2, MB2 = 34, 50, 20, 40
plotW2, plotH2 = W2-ML2-MR2, H2-MT2-MB2
X_MAX = 3.0
Y_MIN, Y_MAX = 0.07, 4.2
def xs(e): return ML2 + plotW2*(e/X_MAX)
def ys(l):
    lo, hi = math.log10(Y_MIN), math.log10(Y_MAX)
    t = (math.log10(l)-lo)/(hi-lo)
    return MT2 + plotH2*(1-t)
out=[]
for tick in (4,2,1,0.5,0.25,0.1):
    y=ys(tick)
    out.append(f'<line class="gl" x1="{ML2}" x2="{W2-MR2}" y1="{y:.1f}" y2="{y:.1f}"/>')
    out.append(f'<text class="ax-y" x="{ML2-8}" y="{y+3:.1f}" text-anchor="end">{tick:g}</text>')
for e in (0,1,2,3):
    x=xs(e)
    out.append(f'<line class="gl-v" x1="{x:.1f}" x2="{x:.1f}" y1="{MT2}" y2="{MT2+plotH2}"/>')
    out.append(f'<text class="ax-x" x="{x:.1f}" y="{H2-14}">epoch {e}</text>')
def series(pts, cls, name, end_epoch, total_steps):
    path = "M " + " L ".join(f"{xs(e):.1f} {ys(l):.1f}" for e,l in pts)
    o=[f'<path class="line {cls}" d="{path}" fill="none"/>']
    for e,l in pts:
        o.append(f'<circle class="dot {cls}" cx="{xs(e):.1f}" cy="{ys(l):.1f}" r="3.6"><title>{name} epoch {e:.2f} loss {l:.3f}</title></circle>')
    # 訓練其實是從 epoch 0 跑到整數終點（108/234 步全部跑完），只是 loss 每 10 步
    # 才記一次，第一筆／最後一筆 log 剛好落在整數之前／之後一小段——起點跟終點都
    # 用虛線＋空心圓標出「訓練實際起訖」，避免看起來提前中斷或憑空冒出來。
    fx, fl = pts[0]
    o.append(f'<line class="line-ext {cls}" x1="{xs(0):.1f}" y1="{ys(fl):.1f}" x2="{xs(fx):.1f}" y2="{ys(fl):.1f}" stroke-dasharray="2 3"/>')
    o.append(f'<circle class="dot-end {cls}" cx="{xs(0):.1f}" cy="{ys(fl):.1f}" r="4.2"><title>{name} 訓練起點：epoch 0</title></circle>')
    lx, ly = pts[-1]
    ex = xs(end_epoch)
    o.append(f'<line class="line-ext {cls}" x1="{lx:.1f}" y1="{ys(ly):.1f}" x2="{ex:.1f}" y2="{ys(ly):.1f}" stroke-dasharray="2 3"/>')
    o.append(f'<circle class="dot-end {cls}" cx="{ex:.1f}" cy="{ys(ly):.1f}" r="4.2"><title>{name} 實際訓練終點：共 {total_steps} 步 / epoch {end_epoch:.0f}</title></circle>')
    o.append(f'<text class="line-end" x="{ex+9:.1f}" y="{ys(ly)+4:.1f}">{name}</text>')
    return "".join(o)
out.append(series(v5_pts, "s-v5", "v5", 2, 108))
out.append(series(v6_pts, "s-v6", "v6", 3, 234))
chart_loss = f'<svg class="chart chart-line" viewBox="0 0 {W2} {H2}" role="img">{"".join(out)}</svg>'
open(SP + r"\slide_chart_loss.html", "w", encoding="utf-8").write(chart_loss)
print("done", len(chart_main), len(chart_ablation), len(chart_loss))
