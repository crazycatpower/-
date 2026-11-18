# -*- coding: utf-8 -*-
"""
組合 posthoc_features.py 算好的特徵，比較幾種「不微調 VLM」的做法。所有參數（門檻、k、權重）都只用 dev 決定，
val57 / ext113 只報分數，不參與任何選擇。

做法：
  vlm          v10 原本的判定（生成的 JSON 有沒有缺失）
  vlm_thr      v10 缺失機率，門檻由 dev 決定
  vlm_thr_cat  同上，但墜落類與其他類各一個門檻
  knn          只看最相似的 k 張已標註照片投票（SigLIP 向量）
  blend        w × v10 機率 + (1-w) × 投票比例，w 與門檻由 dev 決定
  owl_fall     墜落類照片若偵測到護欄／安全網／護蓋等防護設施（分數 ≥ s），判合規；其餘照 v10。s 由 dev 決定
  blend_owl    blend 再加上 owl_fall 的規則

用法：python posthoc_eval.py
"""
import json
import math

import numpy as np

import audit_vlm_common as C

D = C.BASE_DIR / "posthoc"
GUARD_QUERIES = ["guardrail", "safety railing around an opening", "safety net", "stair handrail",
                 "cover plate over a floor opening", "stair tower"]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load():
    sets = json.loads((D / "sets.json").read_text(encoding="utf-8"))
    dev_vlm = json.loads((D / "dev_vlm.json").read_text(encoding="utf-8"))
    for r in sets["dev"]:
        r["p_vlm"] = dev_vlm[r["key"]]
        r["pred_vlm"] = r["p_vlm"] >= 0.5
    emb = np.load(D / "siglip.npz")
    vec = dict(zip(emb["keys"].tolist(), emb["vecs"]))
    owl = json.loads((D / "owl.json").read_text(encoding="utf-8"))
    qi = [owl["queries"].index(q) for q in GUARD_QUERIES]
    for name in ("pool", "dev", "val57", "ext113"):
        for r in sets[name]:
            r["vec"] = vec[f"{name}|{r['key']}"]
            if name != "pool":
                r["guard"] = max(owl["scores"][f"{name}|{r['key']}"][i] for i in qi)
    return sets


def knn_scores(pool, rows, k, tau=0.05):
    P = np.stack([r["vec"] for r in pool])
    y = np.array([float(r["label"]) for r in pool])
    for r in rows:
        sims = P @ r["vec"]
        top = np.argsort(-sims)[:k]
        w = np.exp((sims[top] - sims[top].max()) / tau)
        r["p_knn"] = float((w * y[top]).sum() / w.sum())


def acc(rows, pred):
    return sum(pred(r) == r["label"] for r in rows) / len(rows)


def report(name, rows, pred):
    out = []
    for subset, rs in (("全部", rows), ("墜落", [r for r in rows if r["fall"]]), ("其他", [r for r in rows if not r["fall"]])):
        k = sum(pred(r) == r["label"] for r in rs)
        fp = sum(pred(r) and not r["label"] for r in rs)
        fn = sum(not pred(r) and r["label"] for r in rs)
        lo, hi = wilson(k, len(rs))
        out.append(f"{subset} {k / len(rs):.3f} [{lo:.2f},{hi:.2f}] 誤報{fp:2d} 漏報{fn:2d} (n={len(rs)})")
    return f"  {name:6s} " + " | ".join(out)


def main():
    S = load()
    dev, tests = S["dev"], {k: S[k] for k in ("val57", "ext113")}
    grid = [i / 100 for i in range(20, 91, 2)]
    methods = {}

    methods["vlm"] = lambda r: bool(r["pred_vlm"])

    t = max(grid, key=lambda t: (acc(dev, lambda r: r["p_vlm"] >= t), -abs(t - 0.5)))
    methods["vlm_thr"] = lambda r, t=t: r["p_vlm"] >= t
    print(f"vlm_thr: 門檻 {t}")

    tf = max(grid, key=lambda t: (acc([r for r in dev if r["fall"]], lambda r: r["p_vlm"] >= t), -abs(t - 0.5)))
    to = max(grid, key=lambda t: (acc([r for r in dev if not r["fall"]], lambda r: r["p_vlm"] >= t), -abs(t - 0.5)))
    methods["vlm_thr_cat"] = lambda r, tf=tf, to=to: r["p_vlm"] >= (tf if r["fall"] else to)
    print(f"vlm_thr_cat: 墜落 {tf}、其他 {to}")

    best = None
    for k in (3, 5, 10, 20, 40):
        knn_scores(S["pool"], dev, k)
        for t in grid:
            a = acc(dev, lambda r: r["p_knn"] >= t)
            if best is None or (a, -abs(t - 0.5)) > (best[0], -abs(best[2] - 0.5)):
                best = (a, k, t)
    _, k_best, t_knn = best
    for rows in (dev, *tests.values()):
        knn_scores(S["pool"], rows, k_best)
    methods["knn"] = lambda r, t=t_knn: r["p_knn"] >= t
    print(f"knn: k={k_best} 門檻 {t_knn}（dev {best[0]:.3f}）")

    best = None
    for w in [i / 10 for i in range(0, 11)]:
        for t in grid:
            a = acc(dev, lambda r: w * r["p_vlm"] + (1 - w) * r["p_knn"] >= t)
            if best is None or (a, -abs(t - 0.5), w) > (best[0], -abs(best[2] - 0.5), best[1]):
                best = (a, w, t)
    _, w_best, t_blend = best

    def blend(r, w=w_best, t=t_blend):
        return w * r["p_vlm"] + (1 - w) * r["p_knn"] >= t
    methods["blend"] = blend
    print(f"blend: w={w_best} 門檻 {t_blend}（dev {best[0]:.3f}）")

    s_grid = [i / 100 for i in range(5, 81, 5)] + [9.9]
    s = max(s_grid, key=lambda s: (acc(dev, lambda r: False if (r["fall"] and r["guard"] >= s) else bool(r["pred_vlm"])), s))
    methods["owl_fall"] = lambda r, s=s: False if (r["fall"] and r["guard"] >= s) else bool(r["pred_vlm"])
    s2 = max(s_grid, key=lambda s: (acc(dev, lambda r: False if (r["fall"] and r["guard"] >= s) else blend(r)), s))
    methods["blend_owl"] = lambda r, s=s2: False if (r["fall"] and r["guard"] >= s) else blend(r)
    print(f"owl_fall: 防護設施分數門檻 {s}；blend_owl: {s2}（9.9 表示規則不啟用）")

    for name, rows in (("dev", dev), *tests.items()):
        print(f"\n== {name} ==")
        for m, pred in methods.items():
            print(report(m, rows, pred))


if __name__ == "__main__":
    main()
