# -*- coding: utf-8 -*-
"""
LLM Benchmark Tool — 測試 Ollama 模型效能並產生報告
包含文字測試 + TEST.jpg 圖片稽核（模擬真實 LINE Bot 使用流程）

用法:
  python benchmark.py                              # 全部模型、全部測試
  python benchmark.py --models gemma4:latest       # 指定模型
  python benchmark.py --no-image                  # 跳過圖片測試
  python benchmark.py --image path/to/other.jpg   # 指定圖片
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import openai

# ── 常數 ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
TEST_IMAGE = BASE_DIR / "TEST.jpg"

DEFAULT_MODELS = [
    "gemma4:latest",
]

# 視覺支援：Ollama 上已知支援 vision 的模型關鍵字
VISION_MODEL_KEYWORDS = ["gemma4", "gemma3", "llava", "llama4", "minicpm", "moondream",
                         "bakllava", "cogvlm", "internvl", "phi4", "qwen2-vl", "qwen2.5vl"]

BASE_SYSTEM_PROMPT = (
    "你是專業的工地職安稽核主管，依照交通部公路總局「交通、安衛、環保稽核表」格式，用繁體中文分析施工現場的違規項目。\n"
    "【分工】YOLO 偵測清單（含編號、類別、信心分數、座標、畫面區域）代表模型「看到」的物件與位置，一般情況下請以此為準；"
    "但信心分數偏低（如 <0.4）的框可能是誤判（把雜物、木板、陰影誤認成人員/安全帽/背心等），"
    "務必對照原圖核實，找不到對應視覺證據時不得依該框判定缺失，也不可單獨以低信心框論斷☆立即停工等級的結論。\n"
    "【圍籬≠墜落防止】工地周界圍籬／圍牆是保全設施，不是「墜落防止」項目所指的工作場所邊緣及開口部；"
    "除非圖片能清楚看到地面高低落差、坑洞、樓層開口、屋頂邊緣、鷹架或高處工作平台等具體「高處」構造，"
    "否則不得只因看到圍籬或材料堆在圍籬旁就歸類為墜落防止；找不到任何「高處」證據時，墜落防止留空陣列，"
    "不要為了湊滿類別勉強生出一條。\n"
    "回答必須專業、嚴謹且給予明確的改善建議。\n"
    "【marker說明】☆=有立即危險須停工、※=應即日改善、空字串=五日內改善\n"
    "【result說明】○=符合規定、×=應扣款改善、△=應請改善\n"
    "【項次對照表－務必依此分類，不得跨類別誤用編號】\n"
    "1.01~1.08=墜落防止（皆高風險）｜2.01~2.13=倒塌崩塌防止（2.01~2.10高風險，2.11~2.13一般風險）｜"
    "3.01~3.10=感電防止（3.01~3.05高風險，3.06~3.10一般風險）｜4.01~4.06=火災爆炸（皆高風險）｜"
    "5.01~5.05=中毒缺氧（皆高風險）｜6.01~6.12=交通維持（6.01高風險，6.02~6.12一般風險）。\n"
    "凡不屬於上述1~6類的缺失一律歸入「其他」類別，item 請用「類別名稱-序號」格式（如「環境保護-3」），"
    "不得借用 1~6 類別已使用的數字編號。\n"
    "【語言規定】所有文字內容必須全部使用繁體中文，禁止混入其他語言文字（法規條號、專有名詞除外）。\n"
    "【必填欄位】next_action 與 answer 不得為空字串；regulation_refs 若無合適法規可引用，"
    "請填 [\"無明確對應法規，建議由專業技師依現場實際狀況評估\"]，不得留空陣列。\n"
    "只輸出 JSON，格式嚴格如下：\n"
    '{"summary":"現場概況簡述",'
    '"categories":{'
    '"墜落防止":[{"item":"1.01","marker":"☆","result":"×","description":"缺失說明","location":"地點","deadline":"立即停工"}],'
    '"倒塌崩塌防止":[],"感電防止":[],"火災爆炸":[],"中毒缺氧":[],"交通維持":[],"其他":[]'
    '},'
    '"next_action":"建議改善措施","answer":"詳細法律與技術說明",'
    '"regulation_refs":["條號或出處簡述"]}'
)


# ── 文字測試案例 ───────────────────────────────────────────────────────────
TEXT_TEST_CASES = [
    {
        "id": "short_qa",
        "label": "簡短問答",
        "category": "text",
        "messages": [
            {"role": "user", "content": "工地安全帽的顏色分別代表什麼職位？請簡短說明。"}
        ],
    },
    {
        "id": "safety_report",
        "label": "安全稽核報告撰寫",
        "category": "text",
        "messages": [
            {
                "role": "user",
                "content": (
                    "請根據以下情境生成一份工地安全稽核報告：\n"
                    "- 地點：台北某建案 B2 地下室\n"
                    "- 發現問題：3 名工人未配戴安全帽，電線裸露於潮濕地面，逃生出口遭雜物阻擋\n"
                    "- 日期：2026-05-19\n"
                    "報告需包含風險等級、改善建議與完成期限。"
                ),
            }
        ],
    },
    {
        "id": "regulation_query",
        "label": "法規查詢",
        "category": "text",
        "messages": [
            {
                "role": "user",
                "content": (
                    "依據台灣職業安全衛生法，高架作業（2公尺以上）需要哪些防護措施？"
                    "請列出至少5項具體要求。"
                ),
            }
        ],
    },
    {
        "id": "multi_turn",
        "label": "多輪對話",
        "category": "text",
        "messages": [
            {"role": "user", "content": "什麼是 TBM（工具箱會議）？"},
            {
                "role": "assistant",
                "content": "TBM（Toolbox Meeting）是工地每日開工前舉行的短暫安全會議，通常5-15分鐘，由領班主持，說明當日工作內容、潛在危險與防護措施。",
            },
            {"role": "user", "content": "TBM 應該記錄哪些內容？請用表格呈現。"},
        ],
    },
]


# ── 資料庫脈絡載入 ─────────────────────────────────────────────────────────
def load_db_context() -> str:
    """從 PostgreSQL 拉取近期稽核統計與最新記錄摘要，作為 LLM 背景資訊。"""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return "（資料庫未設定，使用模擬資料）\n近30天共 12 筆稽核，最常見違規：未戴安全帽(5次)、無安全背心(4次)、逃生出口阻擋(3次)。"

    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # 近30天統計
                cur.execute("""
                    SELECT COUNT(*) AS total, COALESCE(SUM(detection_count), 0) AS total_det
                    FROM audit_records
                    WHERE SUBSTR(created_at, 1, 10) >= TO_CHAR(NOW() - INTERVAL '30 days', 'YYYY-MM-DD')
                """)
                stats = cur.fetchone() or {}

                # 最近 3 筆摘要
                cur.execute("""
                    SELECT created_at, summary, detection_count, analysis_mode
                    FROM audit_records
                    ORDER BY created_at DESC
                    LIMIT 3
                """)
                recent = cur.fetchall()

        lines = [f"近30天稽核：共 {stats.get('total', 0)} 筆，偵測物件 {stats.get('total_det', 0)} 個。"]
        if recent:
            lines.append("最近稽核記錄：")
            for r in recent:
                summary_short = (r.get("summary") or "")[:60]
                lines.append(
                    f"  [{r.get('created_at','')[:10]}] {r.get('analysis_mode','')} | "
                    f"偵測{r.get('detection_count',0)}個 | {summary_short}"
                )
        return "\n".join(lines)

    except Exception as e:
        return f"（資料庫讀取失敗: {e}）\n近30天共 12 筆稽核，最常見違規：未戴安全帽(5次)、無安全背心(4次)、逃生出口阻擋(3次)。"


# ── YOLO 偵測 TEST.jpg ────────────────────────────────────────────────────
def run_yolo(image_path: Path) -> tuple[list[dict], str]:
    """
    對 image_path 執行 YOLO 偵測，回傳 (detections, detection_block_text)。
    找不到模型或套件時回傳模擬資料。
    """
    yolo_path = BASE_DIR / "best.pt"
    try:
        import cv2
        from ultralytics import YOLO

        model = YOLO(str(yolo_path) if yolo_path.exists() else "yolov8n.pt")
        results = model.predict(source=str(image_path), conf=0.25, save=False, verbose=False)

        detections = []
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2] if img is not None else (1, 1)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                _REGIONS = (("上左","上中","上右"),("中左","中央","中右"),("下左","下中","下右"))
                region = _REGIONS[min(2, int(cy * 3))][min(2, int(cx * 3))]
                detections.append({
                    "detection_index": len(detections) + 1,
                    "yolo_class": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 3),
                    "region_zh": region,
                    "box_norm": {
                        "x1": round(x1/w, 4), "y1": round(y1/h, 4),
                        "x2": round(x2/w, 4), "y2": round(y2/h, 4),
                    },
                })

    except Exception as e:
        print(f"  [YOLO] 使用模擬偵測資料（{e}）")
        detections = [
            {"detection_index": 1, "yolo_class": "person",  "confidence": 0.91, "region_zh": "中央",
             "box_norm": {"x1": 0.3, "y1": 0.2, "x2": 0.5, "y2": 0.8}},
            {"detection_index": 2, "yolo_class": "no-helmet","confidence": 0.85, "region_zh": "上中",
             "box_norm": {"x1": 0.32,"y1": 0.15,"x2": 0.48,"y2": 0.35}},
            {"detection_index": 3, "yolo_class": "person",  "confidence": 0.78, "region_zh": "中右",
             "box_norm": {"x1": 0.6, "y1": 0.25,"x2": 0.8, "y2": 0.75}},
        ]

    if not detections:
        block = "（YOLO 未偵測到任何物件；請依原圖做一般職安觀察。）"
    else:
        lines = ["[YOLO 偵測清單 — 請逐項對照圖片與法規]"]
        for d in detections:
            bn = d["box_norm"]
            lines.append(
                f"  #{d['detection_index']} 類別={d['yolo_class']} "
                f"信心={d['confidence']:.2f} 區域={d['region_zh']} "
                f"norm_xyxy=({bn['x1']},{bn['y1']})-({bn['x2']},{bn['y2']})"
            )
        block = "\n".join(lines)

    return detections, block


_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB — 超過此大小才需縮圖

def image_to_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    if len(data) > _MAX_IMAGE_BYTES:
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((1024, 1024))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            data = buf.getvalue()
            mime = "image/jpeg"
        except Exception as e:
            print(f"  [benchmark] 圖片壓縮失敗，使用原圖：{e}")
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def model_supports_vision(model: str) -> bool:
    m = model.lower()
    return any(k in m for k in VISION_MODEL_KEYWORDS)


# ── 圖片測試案例產生器 ────────────────────────────────────────────────────
def build_image_test_cases(image_path: Path, db_context: str, detections: list[dict], detection_block: str) -> list[dict]:
    """
    產生 3 個仿真實 LINE Bot 圖片使用場景的測試案例。
    messages 中以 _image_path 標記圖片，由 run_single_test 動態注入 base64。
    """
    modes = [
        {
            "id": "img_audit",
            "label": "照片稽核（主流程）",
            "prompt": (
                "請分析這張施工現場照片的職安合規性。\n\n"
                f"{detection_block}\n\n"
                "[資料庫歷史脈絡]:\n" + db_context
            ),
        },
        {
            "id": "img_risk",
            "label": "風險分析",
            "prompt": (
                "請分析這張照片中的風險、異常與建議處置方式。\n\n"
                f"{detection_block}\n\n"
                "[近期稽核歷史供參考]:\n" + db_context
            ),
        },
        {
            "id": "img_sign",
            "label": "標誌辨識",
            "prompt": (
                "請辨識畫面中的交通標誌或安全標示，說明其意義與注意事項。\n\n"
                f"{detection_block}"
            ),
        },
    ]

    cases = []
    for m in modes:
        cases.append({
            "id": m["id"],
            "label": m["label"],
            "category": "image",
            "detections": detections,
            "detection_count": len(detections),
            "_image_path": str(image_path),
            "_prompt": m["prompt"],
            # messages 由 run_single_test 根據 vision 支援動態建立
            "messages": [{"role": "user", "content": m["prompt"]}],
        })
    return cases


# ── 單一測試執行 ───────────────────────────────────────────────────────────
TIMEOUT_SEC = 120  # 單題最長等待秒數，超過視為失敗


def run_single_test(client: openai.OpenAI, model: str, test: dict) -> dict:
    result = {
        "model": model,
        "test_id": test["id"],
        "test_label": test["label"],
        "category": test.get("category", "text"),
        "detection_count": test.get("detection_count"),
        "vision_used": False,
        "input_messages": [],
        "output": "",
        "error": None,
        "time_to_first_token": None,
        "total_time": None,
        "tokens_prompt": None,
        "tokens_completion": None,
        "tokens_per_sec": None,
    }

    # 組建 messages（圖片案例需動態注入 base64）
    if test.get("category") == "image" and test.get("_image_path"):
        img_path = Path(test["_image_path"])
        prompt_text = test["_prompt"]
        supports_vision = model_supports_vision(model)
        result["vision_used"] = supports_vision

        if supports_vision and img_path.exists():
            data_url = image_to_data_url(img_path)
            user_content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        else:
            user_content = prompt_text  # 純文字 fallback

        messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = test["messages"]

    # 記錄純文字版輸入供報告顯示
    result["input_messages"] = [
        {**m, "content": m["content"] if isinstance(m["content"], str)
         else "[圖片 + 文字]"}
        for m in messages
    ]

    try:
        t_start = time.perf_counter()
        t_first = None
        chunks = []

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.1,
            timeout=TIMEOUT_SEC,
        )
        usage_ref: list = []
        for chunk in stream:
            if t_first is None:
                t_first = time.perf_counter()
                result["time_to_first_token"] = round(t_first - t_start, 3)
            delta = (chunk.choices[0].delta.content if chunk.choices else None) or ""
            if delta:
                chunks.append(delta)
            # 部分端點在最後一個 chunk 回傳 usage
            if getattr(chunk, "usage", None):
                usage_ref.append(chunk.usage)

        t_end = time.perf_counter()
        result["output"] = "".join(chunks)
        result["total_time"] = round(t_end - t_start, 3)

        # prompt tokens：優先使用串流 usage，再退回字元估算（避免額外 API 呼叫）
        if usage_ref and usage_ref[-1].prompt_tokens:
            result["tokens_prompt"] = usage_ref[-1].prompt_tokens
        else:
            prompt_chars = sum(
                len(m["content"]) if isinstance(m["content"], str)
                else sum(p.get("text", "") and len(p["text"]) for p in m["content"] if isinstance(p, dict))
                for m in messages
            )
            result["tokens_prompt"] = max(1, prompt_chars // 4)

        # completion tokens：字元數 / 1.5 是中文 token 的合理估算
        output_text = result["output"]
        result["tokens_completion"] = max(1, len(output_text) // 2) if output_text else 0
        if result["total_time"] and result["tokens_completion"]:
            result["tokens_per_sec"] = round(
                result["tokens_completion"] / result["total_time"], 1
            )

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── 執行全部 Benchmark ────────────────────────────────────────────────────
def run_benchmark(models: list[str], base_url: str, api_key: str,
                  all_cases: list[dict]) -> list[dict]:
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    results = []
    total = len(models) * len(all_cases)
    done = 0

    for model in models:
        print(f"\n{'='*60}")
        print(f"  模型: {model}  (vision={'是' if model_supports_vision(model) else '否'})")
        print(f"{'='*60}")
        for test in all_cases:
            done += 1
            icon = "🖼 " if test.get("category") == "image" else "📝 "
            print(f"  [{done}/{total}] {icon}{test['label']} ... ", end="", flush=True)
            r = run_single_test(client, model, test)
            results.append(r)
            if r["error"]:
                print(f"ERROR: {r['error'][:80]}")
            else:
                vision_tag = " [含圖]" if r["vision_used"] else ""
                print(
                    f"首token {r['time_to_first_token']}s | "
                    f"總計 {r['total_time']}s | "
                    f"~{r['tokens_per_sec']} tok/s{vision_tag}"
                )

    return results


# ── HTML 報告產生器 ────────────────────────────────────────────────────────
def _badge(value, thresholds, labels, colors):
    for threshold, label, color in zip(thresholds, labels, colors):
        if value <= threshold:
            return f'<span class="badge" style="background:{color}">{label}</span>'
    return f'<span class="badge" style="background:{colors[-1]}">{labels[-1]}</span>'


def generate_html(results: list[dict], out_path: Path, image_path: Path | None = None):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = list(dict.fromkeys(r["model"] for r in results))
    tests  = list(dict.fromkeys(r["test_id"] for r in results))
    data   = {(r["model"], r["test_id"]): r for r in results}

    # ── 圖片縮圖（base64 嵌入，不依賴外部路徑）──
    img_tag = ""
    if image_path and image_path.exists():
        img_b64 = base64.b64encode(image_path.read_bytes()).decode()
        img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" class="test-img" alt="TEST image">'

    # ── 摘要表格（分文字 / 圖片兩區段）──
    def summary_section(category_filter, title):
        cat_results = [r for r in results if r.get("category") == category_filter]
        if not cat_results:
            return ""
        rows = ""
        for model in models:
            mr = [r for r in cat_results if r["model"] == model and not r["error"]]
            if not mr:
                continue
            avg_ttft  = round(sum(r["time_to_first_token"] or 0 for r in mr) / len(mr), 3)
            avg_total = round(sum(r["total_time"] or 0 for r in mr) / len(mr), 3)
            avg_tps   = round(sum(r["tokens_per_sec"] or 0 for r in mr) / len(mr), 1)
            errors    = sum(1 for r in results if r["model"] == model
                           and r.get("category") == category_filter and r["error"])
            vision    = "是" if any(r["vision_used"] for r in mr) else "否"
            badge     = _badge(avg_total, [10, 30, 60],
                               ["快", "中", "慢", "很慢"],
                               ["#22c55e", "#eab308", "#f97316", "#ef4444"])
            rows += f"""
            <tr>
              <td><strong>{model}</strong></td>
              <td>{avg_ttft} s</td>
              <td>{avg_total} s {badge}</td>
              <td>{avg_tps} tok/s</td>
              <td>{vision}</td>
              <td>{"<span class='err'>"+str(errors)+"</span>" if errors else "—"}</td>
            </tr>"""
        return f"""
        <h2 class="section">{title}</h2>
        <table>
          <thead><tr><th>模型</th><th>平均首 Token</th><th>平均總耗時</th><th>平均速度</th><th>含圖片</th><th>錯誤</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    summary_html = summary_section("text", "摘要 — 文字測試") + \
                   summary_section("image", "摘要 — 圖片稽核測試（模擬 LINE Bot）")

    # ── 詳細卡片 ──
    def input_display(r: dict) -> str:
        msgs = r.get("input_messages", [])
        last = msgs[-1] if msgs else {}
        content = last.get("content", "")
        if isinstance(content, list):
            content = "[圖片 + 文字 prompt]"
        return content[:400].replace("<", "&lt;") + ("…" if len(content) > 400 else "")

    detail_cards = ""
    for test_id in tests:
        sample = next((r for r in results if r["test_id"] == test_id), {})
        label  = sample.get("test_label", test_id)
        cat    = sample.get("category", "text")
        icon   = "🖼" if cat == "image" else "📝"
        det    = sample.get("detection_count")
        det_tag = f' <span class="det-badge">YOLO: {det} 物件</span>' if det is not None else ""
        detail_cards += f'<h2 class="test-title">{icon} {label}{det_tag}</h2>'

        for model in models:
            r = data.get((model, test_id))
            if not r:
                continue
            if r["error"]:
                body = f'<div class="error-box">⚠ {r["error"]}</div>'
            else:
                vision_tag = ' <span class="vision-tag">含圖片</span>' if r["vision_used"] else ""
                stats = (
                    f'首 token: <strong>{r["time_to_first_token"]}s</strong> | '
                    f'總耗時: <strong>{r["total_time"]}s</strong> | '
                    f'Prompt tokens: <strong>{r["tokens_prompt"] or "—"}</strong> | '
                    f'Output ~tokens: <strong>{r["tokens_completion"] or "—"}</strong> | '
                    f'速度: <strong>{r["tokens_per_sec"] or "—"} tok/s</strong>'
                    f'{vision_tag}'
                )
                output_text = (r["output"] or "").replace("<", "&lt;")
                body = f"""
                <div class="stats">{stats}</div>
                <div class="io-grid">
                  <div class="io-box">
                    <div class="io-label">輸入 Prompt</div>
                    <pre>{input_display(r)}</pre>
                  </div>
                  <div class="io-box">
                    <div class="io-label">模型輸出</div>
                    <pre>{output_text}</pre>
                  </div>
                </div>"""
            detail_cards += f"""
            <div class="card">
              <div class="card-header">{model}</div>
              {body}
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Benchmark — {run_time}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",system-ui,sans-serif;background:#f8fafc;color:#1e293b;padding:2rem}}
  h1{{font-size:1.6rem;margin-bottom:.25rem}}
  .meta{{color:#64748b;font-size:.875rem;margin-bottom:1.5rem}}
  h2.section{{font-size:1.1rem;margin:2rem 0 .75rem;color:#334155;border-left:4px solid #6366f1;padding-left:.6rem}}
  h2.test-title{{font-size:1rem;margin:2.5rem 0 .75rem;color:#475569;display:flex;align-items:center;gap:.4rem}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:1.5rem}}
  th{{background:#6366f1;color:#fff;padding:.65rem 1rem;text-align:left;font-size:.85rem}}
  td{{padding:.6rem 1rem;border-bottom:1px solid #e2e8f0;font-size:.88rem}}
  tr:last-child td{{border-bottom:none}}
  .badge{{border-radius:999px;padding:.15rem .55rem;font-size:.72rem;color:#fff;font-weight:700}}
  .det-badge{{background:#0ea5e9;color:#fff;border-radius:999px;padding:.1rem .45rem;font-size:.72rem;font-weight:700}}
  .vision-tag{{background:#8b5cf6;color:#fff;border-radius:999px;padding:.1rem .45rem;font-size:.72rem;font-weight:700}}
  .err{{color:#ef4444;font-weight:700}}
  .card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:1rem;overflow:hidden}}
  .card-header{{background:#6366f1;color:#fff;padding:.5rem 1rem;font-weight:600;font-size:.9rem}}
  .stats{{padding:.6rem 1rem;font-size:.82rem;color:#475569;border-bottom:1px solid #e2e8f0}}
  .io-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#e2e8f0}}
  .io-box{{background:#fff;padding:.75rem 1rem}}
  .io-label{{font-size:.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-bottom:.35rem}}
  pre{{font-family:"Cascadia Code","Consolas",monospace;font-size:.78rem;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow-y:auto;color:#334155}}
  .error-box{{padding:.75rem 1rem;color:#ef4444;font-size:.875rem}}
  .test-img{{max-width:320px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);margin-bottom:1rem;display:block}}
  .img-section{{background:#fff;border-radius:10px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:1.5rem;display:flex;align-items:flex-start;gap:1.5rem}}
  .img-meta{{font-size:.85rem;color:#475569;line-height:1.7}}
  @media(max-width:700px){{.io-grid{{grid-template-columns:1fr}}.img-section{{flex-direction:column}}}}
</style>
</head>
<body>
<h1>LLM Benchmark Report</h1>
<div class="meta">產生時間：{run_time} ｜ 測試模型：{len(models)} 個 ｜ 測試案例：{len(tests)} 項</div>

{"<div class='img-section'>" + img_tag + "<div class='img-meta'><strong>測試圖片：TEST.jpg</strong><br>此圖片用於模擬 LINE Bot 使用者傳送施工現場照片的真實情境。<br>系統流程：上傳圖片 → YOLO 偵測 → 組合 DB 歷史脈絡 → LLM 分析。<br>支援視覺的模型會同時傳入圖片與文字 prompt；不支援的模型僅傳文字。</div></div>" if img_tag else ""}

{summary_html}

<h2 class="section">詳細結果</h2>
{detail_cards}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


# ── Tool-use Benchmark ────────────────────────────────────────────────────────

TOOL_TEST_CASES = [
    {
        "id": "tool_reg_query",
        "label": "法規查詢（含 Tool Use）",
        "question": "依據台灣職業安全衛生法，高架作業（2公尺以上）需要哪些防護措施？請列出至少5項具體要求並附條號。",
    },
    {
        "id": "tool_audit_history",
        "label": "稽核歷史分析（含 Tool Use）",
        "question": "請查詢近30天的稽核記錄，說明最常見的違規類型與改善建議。",
    },
    {
        "id": "tool_combined",
        "label": "綜合查詢（法規 + 歷史）",
        "question": "根據近期稽核記錄，現場最常出現安全帽相關違規，請查詢相關法規並給出具體改善方案。",
    },
]

TOOL_SYSTEM_PROMPT = (
    "你是專業的工地職安稽核主管，專長是用繁體中文分析施工現場的違規項目。\n"
    "你有以下工具可以使用：\n"
    "1. search_regulations：搜尋台灣職業安全衛生法規資料庫，取得精確條文\n"
    "2. query_audit_history：查詢現場稽核記錄與統計\n"
    "回答前請先呼叫適當工具取得資料，再根據實際資料庫內容回答，不要憑空捏造條號。"
)


def run_tool_benchmark(models: list[str], base_url: str, api_key: str) -> list[dict]:
    from tools_chat import chat_with_tools

    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    results = []
    total = len(models) * len(TOOL_TEST_CASES)
    done = 0

    for model in models:
        print(f"\n{'='*60}")
        print(f"  模型: {model}  [Tool Use 測試]")
        print(f"{'='*60}")
        for case in TOOL_TEST_CASES:
            done += 1
            print(f"  [{done}/{total}] {case['label']} ... ", end="", flush=True)
            messages = [{"role": "user", "content": case["question"]}]
            r = chat_with_tools(
                client, model, messages,
                system_prompt=TOOL_SYSTEM_PROMPT,
                max_turns=5,
                timeout=120,
            )
            n_calls = len(r["tool_calls_made"])
            if r["error"]:
                print(f"ERROR: {r['error'][:80]}")
            else:
                tools_used = ", ".join(c["tool"] for c in r["tool_calls_made"]) or "（未呼叫工具）"
                print(f"工具呼叫 {n_calls} 次 [{tools_used}] | 總計 {r['total_time']}s")
            results.append({
                "model": model,
                "test_id": case["id"],
                "test_label": case["label"],
                "question": case["question"],
                "output": r["output"],
                "tool_calls_made": r["tool_calls_made"],
                "turns_used": r["turns_used"],
                "total_time": r["total_time"],
                "error": r["error"],
            })

    return results


def generate_tool_html(results: list[dict], out_path: Path):
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = list(dict.fromkeys(r["model"] for r in results))

    rows = ""
    for r in results:
        calls = r.get("tool_calls_made") or []
        n = len(calls)
        called = "✓" if n > 0 else "✗"
        tool_names = "<br>".join(f"<code>{c['tool']}({list(c['args'].values())[0] if c['args'] else ''})</code>" for c in calls) or "—"
        output_escaped = (r.get("output") or r.get("error") or "").replace("<", "&lt;")[:800]
        rows += f"""
        <tr>
          <td><strong>{r['model']}</strong></td>
          <td>{r['test_label']}</td>
          <td style="text-align:center;font-size:1.2rem">{'<span style="color:#22c55e">✓</span>' if n>0 else '<span style="color:#ef4444">✗</span>'}</td>
          <td>{tool_names}</td>
          <td>{r.get('turns_used','—')}</td>
          <td>{r.get('total_time','—')} s</td>
          <td><pre style="max-height:200px;overflow-y:auto;font-size:.75rem;white-space:pre-wrap">{output_escaped}</pre></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>Tool Use Benchmark — {run_time}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",system-ui,sans-serif;background:#f8fafc;color:#1e293b;padding:2rem}}
  h1{{font-size:1.5rem;margin-bottom:.25rem}}
  .meta{{color:#64748b;font-size:.85rem;margin-bottom:1.5rem}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
  th{{background:#6366f1;color:#fff;padding:.65rem 1rem;text-align:left;font-size:.82rem}}
  td{{padding:.6rem 1rem;border-bottom:1px solid #e2e8f0;font-size:.82rem;vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  code{{background:#f1f5f9;padding:.1rem .3rem;border-radius:4px;font-size:.78rem}}
  pre{{font-family:"Cascadia Code",monospace;font-size:.75rem;white-space:pre-wrap;word-break:break-word}}
</style>
</head>
<body>
<h1>Tool Use Benchmark Report</h1>
<div class="meta">產生時間：{run_time} ｜ 測試模型：{len(models)} 個 ｜ 測試案例：{len(TOOL_TEST_CASES)} 項</div>
<table>
  <thead>
    <tr>
      <th>模型</th><th>測試案例</th><th>呼叫工具</th><th>工具名稱</th><th>對話輪次</th><th>耗時</th><th>最終回答</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ── 主程式 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ollama LLM Benchmark Tool")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--image", default=str(TEST_IMAGE), help="測試圖片路徑")
    parser.add_argument("--no-image", action="store_true", help="跳過圖片測試")
    parser.add_argument("--no-text", action="store_true", help="跳過文字測試")
    parser.add_argument("--tools", action="store_true", help="執行 Tool Use 測試（模型主動查 DB）")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    base_url  = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key   = os.getenv("LLM_API_KEY", "ollama")
    env_model = os.getenv("LLM_MODEL", "")

    models = args.models or DEFAULT_MODELS
    if env_model and env_model not in models:
        models = [env_model] + models
    models = list(dict.fromkeys(models))

    # ── 組合所有測試案例 ──
    all_cases: list[dict] = []

    if not args.no_text:
        all_cases += TEXT_TEST_CASES

    image_path = None
    if not args.no_image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"⚠ 找不到圖片 {image_path}，跳過圖片測試")
        else:
            print(f"\n載入圖片: {image_path}")
            print("執行 YOLO 偵測 ...", end=" ", flush=True)
            detections, detection_block = run_yolo(image_path)
            print(f"偵測到 {len(detections)} 個物件")

            print("讀取資料庫脈絡 ...", end=" ", flush=True)
            db_context = load_db_context()
            print("完成")

            all_cases += build_image_test_cases(image_path, db_context, detections, detection_block)

    print(f"\nOllama endpoint : {base_url}")
    print(f"測試模型        : {', '.join(models)}")
    print(f"測試案例數      : 文字 {sum(1 for c in all_cases if c.get('category')=='text')} | "
          f"圖片 {sum(1 for c in all_cases if c.get('category')=='image')}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # ── Tool Use 模式 ──
    if args.tools:
        print(f"\nOllama endpoint : {base_url}")
        print(f"測試模型        : {', '.join(models)}")
        print(f"模式            : Tool Use（模型主動查 DB）")
        tool_results = run_tool_benchmark(models, base_url, api_key)

        html_path = Path(args.output) if args.output else out_dir / f"toolbench_{ts}.html"
        generate_tool_html(tool_results, html_path)
        print(f"\n報告已儲存 → {html_path.resolve()}")

        json_path = Path(args.json) if args.json else out_dir / f"toolbench_{ts}.json"
        json_path.write_text(json.dumps(tool_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"原始資料       → {json_path.resolve()}")
        return

    # ── 一般 Benchmark 模式 ──
    results = run_benchmark(models, base_url, api_key, all_cases)

    html_path = Path(args.output) if args.output else out_dir / f"benchmark_{ts}.html"
    generate_html(results, html_path, image_path)
    print(f"\n報告已儲存 → {html_path.resolve()}")

    json_path = Path(args.json) if args.json else out_dir / f"benchmark_{ts}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原始資料       → {json_path.resolve()}")


if __name__ == "__main__":
    main()
