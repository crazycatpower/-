# -*- coding: utf-8 -*-
"""組合 report_line_text.json + 圖片 base64 產生純文件版面（非手機介面）報告。"""
import json
import html as html_mod
from pathlib import Path

BASE = Path(__file__).parent
texts = json.loads((BASE / "report_line_text.json").read_text(encoding="utf-8"))
images = json.loads((BASE / "scratch_images_b64.json").read_text(encoding="utf-8"))

CATS = [
    {"key": "01", "num": "1", "name": "墜落防止", "photo": "施工架高處作業"},
    {"key": "02", "num": "2", "name": "倒塌崩塌防止", "photo": "深基坑開挖工程"},
    {"key": "03", "num": "3", "name": "感電防止", "photo": "工業電纜與插頭"},
    {"key": "04", "num": "4", "name": "火災爆炸", "photo": "鋼構電焊作業"},
    {"key": "05", "num": "5", "name": "中毒缺氧", "photo": "深井/坑道作業"},
    {"key": "06", "num": "6", "name": "交通維持", "photo": "道路鋪設工程"},
]

RISK_CLASS = {"🔴": "risk-high", "🟠": "risk-med", "🟡": "risk-low", "🟢": "risk-low"}


def esc(x):
    return html_mod.escape(str(x or ""))


def render_section(cat, text):
    risk_cls = RISK_CLASS.get(text[:1], "risk-low")
    img_b64 = images[cat["key"]]
    return f"""
  <section class="record">
    <div class="record-media">
      <img src="data:image/jpeg;base64,{img_b64}" alt="{esc(cat['photo'])}" loading="lazy">
      <div class="media-caption">{esc(cat['photo'])}</div>
    </div>
    <div class="record-body">
      <div class="record-head">
        <span class="num-badge {risk_cls}">{cat['num']}</span>
        <div class="record-title">
          <div class="eyebrow">類別 {cat['num']}</div>
          <div class="name">{esc(cat['name'])}</div>
        </div>
      </div>
      <span class="source-label">build_line_analysis_reply() 原始輸出</span>
      <div class="raw-reply">{esc(text)}</div>
    </div>
  </section>"""


sections_html = "".join(
    render_section(cat, item["text"]) for cat, item in zip(CATS, texts)
)

template = (BASE / "plain_report_template.html").read_text(encoding="utf-8")
out = template.replace("__SECTIONS__", sections_html)

out_path = BASE / "plain_audit_report.html"
out_path.write_text(out, encoding="utf-8")
print("wrote", out_path, len(out), "chars")
