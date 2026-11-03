# -*- coding: utf-8 -*-
"""組合 report_line_text.json + 圖片 base64 產生「像在 LINE 看到的」聊天畫面 HTML。"""
import json
import html as html_mod
from pathlib import Path

BASE = Path(__file__).parent
texts = json.loads((BASE / "report_line_text.json").read_text(encoding="utf-8"))
images = json.loads((BASE / "scratch_images_b64.json").read_text(encoding="utf-8"))

CATS = [
    {"key": "01", "num": "1", "name": "墜落防止"},
    {"key": "02", "num": "2", "name": "倒塌崩塌防止"},
    {"key": "03", "num": "3", "name": "感電防止"},
    {"key": "04", "num": "4", "name": "火災爆炸"},
    {"key": "05", "num": "5", "name": "中毒缺氧"},
    {"key": "06", "num": "6", "name": "交通維持"},
]

TIMES = ["14:02", "14:09", "14:15", "14:23", "14:31", "14:40"]


def esc(x):
    return html_mod.escape(str(x or ""))


def render_exchange(cat, text, t):
    img_b64 = images[cat["key"]]
    return f"""
    <div class="divider">類別 {cat['num']}・{esc(cat['name'])}</div>

    <div class="row user">
      <div class="meta"><span class="read">已讀</span><span>{t}</span></div>
      <div class="img-bubble"><img src="data:image/jpeg;base64,{img_b64}" alt="使用者傳送的照片" loading="lazy"></div>
    </div>

    <div class="row bot">
      <div class="bot-avatar">🦺</div>
      <div class="bubble notice">🔎 已收到照片，正在進行 AI 職安稽核分析，請稍候...</div>
      <div class="meta">{t}</div>
    </div>

    <div class="row bot">
      <div class="bot-avatar spacer">🦺</div>
      <div class="bubble">{esc(text)}</div>
      <div class="meta">{t}</div>
    </div>
"""


messages_html = "".join(
    render_exchange(cat, item["text"], t)
    for cat, item, t in zip(CATS, texts, TIMES)
)

template = (BASE / "line_report_template.html").read_text(encoding="utf-8")
out = template.replace("__MESSAGES__", messages_html)

out_path = BASE / "line_chat_preview.html"
out_path.write_text(out, encoding="utf-8")
print("wrote", out_path, len(out), "chars")
