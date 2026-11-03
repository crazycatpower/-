# -*- coding: utf-8 -*-
"""一次性腳本：組合 report_data.json + 圖片 base64 產生簡報 HTML。"""
import json
import html as html_mod
from pathlib import Path
from audit_checklist import official_text

BASE = Path(__file__).parent
data = json.loads((BASE / "report_data.json").read_text(encoding="utf-8"))
images = json.loads((BASE / "scratch_images_b64.json").read_text(encoding="utf-8"))

CATS = [
    {"key": "01", "num": "1", "name": "墜落防止", "photo": "施工架高處作業"},
    {"key": "02", "num": "2", "name": "倒塌崩塌防止", "photo": "深基坑開挖工程"},
    {"key": "03", "num": "3", "name": "感電防止", "photo": "工業電纜與插頭"},
    {"key": "04", "num": "4", "name": "火災爆炸", "photo": "鋼構電焊作業"},
    {"key": "05", "num": "5", "name": "中毒缺氧", "photo": "深井/坑道作業"},
    {"key": "06", "num": "6", "name": "交通維持", "photo": "道路鋪設工程"},
]

RISK_CLASS = {"高": "risk-high", "中": "risk-med", "低": "risk-low"}


def esc(x):
    return html_mod.escape(str(x or ""))


def render_defect(d):
    rule = official_text(d["item"]) or ""
    marker = d.get("marker") or ""
    marker_html = f'<span class="marker">{esc(marker)}</span>' if marker else ""
    return f"""
    <li class="defect">
      <div class="defect-head">
        <span class="item-code">{esc(d['item'])}</span>
        {marker_html}
        <span class="deadline">{esc(d.get('deadline',''))}</span>
      </div>
      {f'<p class="rule-quote">{esc(rule)}</p>' if rule else ''}
      <p class="observation">{esc(d.get('desc',''))}</p>
      <p class="loc">📍 {esc(d.get('loc',''))}</p>
    </li>"""


def render_section(cat, record):
    risk_cls = RISK_CLASS.get(record["risk_label"], "risk-low")
    defects = record["defects"]
    defects_html = "".join(render_defect(d) for d in defects) if defects else '<p class="clear">本次分析未發現缺失項目。</p>'
    refs = record.get("regulation_refs") or []
    refs_html = "、".join(esc(r) for r in refs)
    img_b64 = images[cat["key"]]

    return f"""
  <section class="record" id="cat-{cat['key']}">
    <div class="record-media">
      <img src="data:image/jpeg;base64,{img_b64}" alt="{esc(cat['photo'])}" loading="lazy">
      <div class="media-caption">{esc(cat['photo'])}</div>
    </div>
    <div class="record-body">
      <div class="record-head">
        <span class="num-badge {risk_cls}">{cat['num']}</span>
        <div class="record-title">
          <div class="eyebrow">類別 {cat['num']} · {esc(cat['name'])}</div>
          <div class="risk-row">
            <span class="risk-pill {risk_cls}">風險等級：{esc(record['risk_label'])}</span>
            <span class="det-count">YOLO 偵測 {record['det_count']} 項</span>
            <span class="defect-count">缺失 {len(defects)} 項</span>
          </div>
        </div>
      </div>
      <p class="summary">{esc(record['summary'])}</p>
      <ul class="defect-list">
        {defects_html}
      </ul>
      <div class="next-action">
        <div class="label">改善建議</div>
        <p>{esc(record['next_action'])}</p>
      </div>
      <div class="refs">法規依據：{refs_html}</div>
    </div>
  </section>"""


def render_summary_card(cat, record):
    risk_cls = RISK_CLASS.get(record["risk_label"], "risk-low")
    return f"""
      <a class="sum-card {risk_cls}" href="#cat-{cat['key']}">
        <span class="sum-num">{cat['num']}</span>
        <span class="sum-name">{esc(cat['name'])}</span>
        <span class="sum-risk">{esc(record['risk_label'])}</span>
        <span class="sum-defects">{len(record['defects'])} 項缺失</span>
      </a>"""


records_by_key = {c["key"]: r for c, r in zip(CATS, data)}
summary_html = "".join(render_summary_card(c, records_by_key[c["key"]]) for c in CATS)
sections_html = "".join(render_section(c, records_by_key[c["key"]]) for c in CATS)

high_n = sum(1 for r in data if r["risk_label"] == "高")
med_n = sum(1 for r in data if r["risk_label"] == "中")
low_n = sum(1 for r in data if r["risk_label"] == "低")
total_defects = sum(len(r["defects"]) for r in data)

template = (BASE / "report_template.html").read_text(encoding="utf-8")
out = (
    template
    .replace("__SUMMARY_CARDS__", summary_html)
    .replace("__SECTIONS__", sections_html)
    .replace("__HIGH_N__", str(high_n))
    .replace("__MED_N__", str(med_n))
    .replace("__LOW_N__", str(low_n))
    .replace("__TOTAL_DEFECTS__", str(total_defects))
)

out_path = BASE / "audit_test_report.html"
out_path.write_text(out, encoding="utf-8")
print("wrote", out_path, len(out), "chars")
