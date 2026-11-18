# -*- coding: utf-8 -*-
"""比較各模型在「墜落防止（大類 1）」照片上的表現。墜落照片依答案表 B 欄（大類編號）= 1 判定。

用法：python compare_fall.py results/ext113_xxx.json results/val57_yyy.json ...
"""
import json
import sys
from pathlib import Path

import openpyxl

import audit_vlm_common as C


def fall_names(xlsx, sheet=0):
    wb = openpyxl.load_workbook(xlsx)
    ws = wb.worksheets[sheet]
    return {str(r[0]).replace("\\", "/").split("/")[-1] for r in ws.iter_rows(values_only=True)
            if r[0] and r[1] is not None and str(r[1]).strip() in ("1", "1.0")}


FALL = fall_names(C.ANSWER_XLSX) | fall_names(C.EXT_XLSX)


def metrics(rows):
    c = {k: sum(r["status"] == k for r in rows) for k in ("TP", "TN", "FP", "FN", "PARSE_ERROR")}
    n = len(rows)
    acc = (c["TP"] + c["TN"]) / n if n else 0
    return f"n={n:3d} acc={acc:.3f} TP={c['TP']:2d} TN={c['TN']:2d} FP={c['FP']:2d} FN={c['FN']:2d} 壞={c['PARSE_ERROR']}"


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        fall = [r for r in rows if Path(r["image"]).name in FALL]
        other = [r for r in rows if Path(r["image"]).name not in FALL]
        print(f"{Path(path).stem}\n  墜落  {metrics(fall)}\n  其他  {metrics(other)}\n  全部  {metrics(rows)}")
