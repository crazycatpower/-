import argparse
import base64
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from ultralytics import YOLO

from knowledge_base import format_snippets, retrieve_regulations
from audit_checklist import (
    ALL_KNOWN_CODES,
    CHECKLIST_1_TO_6,
    CODE_TO_EXPECTED_CATEGORY,
    official_marker_and_deadline,
    render_checklist_1_to_6,
    render_other_reference_items,
)

# 職安相關關鍵字，確保 YOLO 偵測到的這些物件會被優先分析
DEFAULT_KEYWORDS = ["person", "helmet", "vest", "gloves", "boots", "safety", "hard-hat", "no-helmet"]

# 低於此信心分數的 YOLO 偵測會在 prompt 中標記為「低信心」，避免 LLM 把雜訊誤判
# （如把斜靠的木板/雜物看成人員）當成鐵證，腦補出未實際發生的高風險缺失。
YOLO_LOW_CONF_THRESHOLD = float(os.getenv("YOLO_LOW_CONF_THRESHOLD", "0.4"))

# --- 切片（tiling / SAHI）偵測設定 ---
# 大圖直接整張丟給 YOLO 時，遠處的小物件（遠方工人、小型安全裝備）容易漏偵測。
# 切片偵測會把大圖切成有重疊的小塊、逐塊跑 YOLO、再用 NMS 合併，明顯提升小物件召回率。
# 為了避免小圖也付出多次推論的代價，預設只在「長邊 > YOLO_SLICE_TRIGGER_SIDE」的大圖才啟用。
YOLO_USE_SLICING = os.getenv("YOLO_USE_SLICING", "1").strip() not in ("0", "false", "False", "no")
# 圖片長邊超過此像素才切片；小於此值維持單次整張推論。
# 模型輸入 imgsz=640，長邊 >960 的圖被縮到 640 時遠處小物件已明顯損失，故從 960 起切片。
YOLO_SLICE_TRIGGER_SIDE = int(os.getenv("YOLO_SLICE_TRIGGER_SIDE", "960"))
# 每個切片的寬高（像素）。SAHI 會依圖片大小自動算出要切幾塊，不需手動指定塊數。
YOLO_SLICE_SIZE = int(os.getenv("YOLO_SLICE_SIZE", "512"))
# 相鄰切片的重疊比例（0~1），避免物件剛好被切在邊界上而漏偵測。
YOLO_SLICE_OVERLAP = float(os.getenv("YOLO_SLICE_OVERLAP", "0.2"))
# 切片專用的最小信心門檻。切片會把雜物（鋼筋切面、樹葉）放大，容易誤判成小物件，
# 故預設用比整張推論（min_conf 0.25）更嚴的 0.3 過濾明顯雜訊；可依現場調整。
YOLO_SLICE_MIN_CONF = float(os.getenv("YOLO_SLICE_MIN_CONF", "0.3"))

BASE_SYSTEM_PROMPT = (
    "你是專業的工地職安稽核主管，依照交通部公路總局「交通、安衛、環保稽核表」格式，用繁體中文分析施工現場的違規項目。\n"
    "【分工】YOLO 偵測清單（含編號、類別、信心分數、座標、畫面區域）代表模型「看到」的物件與位置，一般情況下請以此為準；"
    "圖片（原圖與標註圖）用於補充情境、細節與未列入清單的環境風險。\n"
    "【低信心框】清單中標記⚠️低信心的框，代表 YOLO 本身也不確定（可能把雜物、木板、陰影誤判成人員/安全帽/背心等），"
    "務必對照原圖該座標範圍實際核實：若原圖該處看不出對應的人員或物件，就不能依該框判定缺失，"
    "更不可單獨以低信心框作為「墜落防止」等☆立即停工等級結論的依據；找不到高處作業、開口、鷹架、平台等實際"
    "「高處」構造證據時，不得憑空假設有人在高處作業。\n"
    "【圍籬≠墜落防止】工地周界的圍籬／圍牆是保全、隔離用設施，不是稽核表「墜落防止」項目所指的「工作場所邊緣"
    "及開口部」；除非圖片中能清楚看到地面本身有高低落差、坑洞、樓層開口、屋頂邊緣、鷹架或高處工作平台等具體"
    "「高處」構造，否則不得只因看到圍籬、或材料堆放在圍籬旁，就把它當成墜落防止的缺失。物料堆放方式、動線、"
    "整潔等問題請歸類到「其他」（如物體飛落防止、一般性規定），不要塞進「墜落防止」硬套模板。若整張圖找不到"
    "任何「高處」構造證據，「墜落防止」底下每一條 1.01~1.08 都應判定為 ○（符合規定/未發現異常），"
    "不得為了湊出缺失而勉強把某一條判定為×或△。\n"
    "【標註圖】方框為 YOLO 偵測位置，請與清單中的 detection_index、box_norm（0~1 相對座標）互相對照。\n"
    "請以「職安第一」為原則，引用下方【參考法規】片段（若有）推論合規性；無法從片段確定時，請說明不確定處，勿捏造條號。\n"
    "回答必須專業、嚴謹且給予明確的改善建議。\n"
    "【marker說明】☆=勞動檢查法第28條有立即危險之虞須停工改善、※=應即日改善、空字串=五日內改善\n"
    "【result說明】○=符合規定、×=應扣款改善（缺失）、△=應請改善（待改善）\n"
    "【deadline說明】☆對應「立即停工」、※對應「即日改善」、其餘填「五日內改善」\n"
    "【確定性與 marker 一致性】☆代表你能在圖片中明確指出具體證據（實際看到人員在高處、實際看到未防護的開口/邊緣等）。"
    "若你自己的描述包含「疑似」「無法確認」「看不清楚」「可能」等不確定用語，代表證據不足，"
    "該項目最高只能標※，且應在 description 中明確寫出不確定之處；證據充分時才可標☆。"
    "不得一邊承認無法確認、一邊標記☆。\n"
    "【嚴格依據官方條文、逐條打勾】下方會提供【官方稽核項次清單（第1~6類，共54條）】與【一般風險／環境保護"
    "參考清單】，文字逐字取自「05招標規範附件.pdf」，是唯一合法的項次來源。\n"
    "categories 中「墜落防止、倒塌崩塌防止、感電防止、火災爆炸、中毒缺氧、交通維持」這6個類別，"
    "都必須像勾稽核表一樣，把該類別清單中的每一條都列出來、逐條給出 result（○=符合規定/未發現異常、"
    "×=缺失、△=待改善），不可以只挑幾條列、其餘省略——這6類加起來剛好54條，每一條都要出現一次，"
    "不多不少。只有 result 不是○的項目才需要填 description（具體觀察到的缺失內容）與 location；"
    "result是○的項目，description/location可以省略或留空字串。marker與deadline由系統依官方清單自動帶入，"
    "你不需要自己判斷，但仍請照清單填上（填錯也會被系統覆蓋修正）。item 欄位務必填清單中真實存在的代碼"
    "（如 1.01、3.07），絕對禁止自創代碼、絕對禁止把某個代碼判到不屬於它的類別裡。\n"
    "第1~6類（1.01~6.12）以外若有觀察到的缺失，一律歸入「其他」類別，item 代碼須取自【一般風險／環境保護"
    "參考清單】（如「7.01」「環保-3」「其他不符-1」），且 description 內容務必是該代碼官方條文實際描述的情境"
    "（例如 7.01 是講機械操作半徑管制，不能拿來寫電纜線亂放）；「其他」不需要逐條列出，只列有觀察到的缺失即可，"
    "其值仍必須是陣列，不可把「其他」本身變成物件或巢狀 key。\n"
    "item 編號須嚴格對應官方清單，不得跨類別誤用編號。\n"
    "【語言規定】summary、description、next_action、answer 等所有文字內容必須全部使用繁體中文，"
    "禁止混入任何其他語言文字（含英文單字、日文、韓文、泰文等），法規條號、專有名詞除外。\n"
    "【必填欄位】next_action 與 answer 不得為空字串；regulation_refs 若無合適法規可引用，"
    "請填 [\"無明確對應法規，建議由專業技師依現場實際狀況評估\"]，不得留空陣列。\n"
    "【regulation_refs 只能是真正的外部法規】regulation_refs 必須引用真實存在的法規名稱與條號"
    "（如《職業安全衛生法》第幾條、《營造安全衛生設施標準》第幾條），並優先引用【參考法規】片段"
    "（若有提供）；絕對不可以把本系統自己的稽核表項次代碼（如「1.02」「10.02」「環保-3」）當成"
    "法規來源填進 regulation_refs——項次代碼只是用來對照 categories 裡的缺失項目，不是法規本身。\n"
    "只輸出 JSON，物件內每個 item 的格式如下（categories 底下 6 個類別各自的陣列，"
    "務必包含該類別官方清單的每一條，一條一個物件，不得省略任何一條）：\n"
    '{"summary":"現場概況簡述",'
    '"categories":{'
    '"墜落防止":[{"item":"1.01","result":"○"},{"item":"1.02","result":"×","description":"缺失說明","location":"缺失地點"}],'
    '"倒塌崩塌防止":[{"item":"2.01","result":"○"}],'
    '"感電防止":[{"item":"3.01","result":"○"}],'
    '"火災爆炸":[{"item":"4.01","result":"○"}],'
    '"中毒缺氧":[{"item":"5.01","result":"○"}],'
    '"交通維持":[{"item":"6.01","result":"○"}],'
    '"其他":[]'
    '},'
    '"next_action":"建議改善措施",'
    '"answer":"詳細法律與技術說明",'
    '"regulation_refs":["《職業安全衛生法》第OO條（或參考法規片段中的實際條號）"],'
    '"by_detection":[{"detection_index":1,"note":"與該框相關之風險或合規說明（無則留空字串）"}]}'
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="偵測工地物件並進行職安風險分析")
    parser.add_argument("--image", required=True, help="輸入圖片路徑")
    parser.add_argument(
        "--yolo-model",
        default="best.pt",
        help="YOLO 權重路徑（預設專案根目錄 best.pt；相對路徑以專案根為準）",
    )
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM 模型名稱")
    parser.add_argument("--output-dir", default="outputs", help="輸出資料夾")
    parser.add_argument("--min-conf", type=float, default=0.25, help="最小信心分數")
    return parser.parse_args()

def build_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    # 送進 LLM 前先縮圖：過大的原圖（如手機 5712×4284）會讓小型 vision 模型
    # （gemma4:e2b）在 json_object 約束下回傳空字串，且額外吃 VRAM、拖慢推論。
    image_bytes = _downscale_for_llm(image_bytes)
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"


def _downscale_for_llm(image_bytes: bytes, max_side: int = 1024) -> bytes:
    """將圖片長邊縮到 max_side 以內；失敗則回傳原始 bytes。"""
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) <= max_side:
            return image_bytes
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        print(f"[LLM] 圖片縮放失敗，使用原圖: {e}")
        return image_bytes

# 以框中心落在 3×3 九宮格描述位置（相對整張圖）
_REGION_LABELS_3X3 = (
    ("上左", "上中", "上右"),
    ("中左", "中央", "中右"),
    ("下左", "下中", "下右"),
)


def enrich_detections_spatial(detections: list[dict], width: int, height: int) -> None:
    """就地補上 detection_index、region_zh、box_norm（相對 0~1）。"""
    if width <= 0 or height <= 0:
        return
    for i, d in enumerate(detections):
        d["detection_index"] = i + 1
        x1, y1, x2, y2 = d["box"]
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        col = min(2, max(0, int(cx * 3)))
        row = min(2, max(0, int(cy * 3)))
        d["region_zh"] = _REGION_LABELS_3X3[row][col]
        d["box_norm"] = {
            "x1": round(x1 / width, 4),
            "y1": round(y1 / height, 4),
            "x2": round(x2 / width, 4),
            "y2": round(y2 / height, 4),
        }


def format_detections_for_prompt(detections: list[dict]) -> str:
    """給 LLM 的結構化偵測段落（位置 + 類別）。信心過低的框會加註警告，
    提醒 LLM 這可能是誤判（如把雜物/木板誤認成人員），不能單獨當成鐵證。
    """
    if not detections:
        return "（無 YOLO 偵測框；請依原圖做一般職安觀察，並註明無偵測資料。）"
    lines: list[str] = []
    low_conf_count = 0
    for d in detections:
        idx = d.get("detection_index", "?")
        cls = d.get("yolo_class", "")
        conf = float(d.get("confidence", 0))
        region = d.get("region_zh", "")
        bn = d.get("box_norm") or {}
        box_s = (
            f"norm_xyxy=({bn.get('x1')},{bn.get('y1')})-({bn.get('x2')},{bn.get('y2')})"
            if bn
            else "norm_xyxy=(n/a)"
        )
        is_low_conf = conf < YOLO_LOW_CONF_THRESHOLD
        if is_low_conf:
            low_conf_count += 1
        flag = "  ⚠️低信心，可能為誤判" if is_low_conf else ""
        lines.append(
            f"  #{idx} 類別={cls} 信心={conf:.2f} 區域={region} {box_s} "
            f"像素_xyxy={d.get('box')}{flag}"
        )
    header = "[YOLO 偵測清單 — 請逐項對照圖片與法規]"
    if low_conf_count:
        header += (
            f"\n（註：信心 <{YOLO_LOW_CONF_THRESHOLD:.2f} 已標記⚠️低信心，"
            f"本清單中有 {low_conf_count} 個低信心框；務必以原圖實際內容核實後再引用，"
            "不得單獨依據低信心框判定缺失，尤其不可用它作為☆立即停工等級結論的唯一依據。）"
        )
    return header + "\n" + "\n".join(lines)


def build_regulation_retrieval_query(prompt: str, detections: list[dict]) -> str:
    """法規片段檢索：合併使用者意圖與偵測類別，提升關鍵字命中。"""
    base = (prompt or "").strip()
    classes = sorted({str(d.get("yolo_class", "")).strip() for d in detections if d.get("yolo_class")})
    if not classes:
        return base
    return f"{base}\n偵測類別：{'、'.join(classes)}"


# schema 規定為 categories 的同層欄位；若因輸出截斷導致誤塞進 categories 內，需搬回上層
_TOP_LEVEL_KEYS = ("next_action", "answer", "regulation_refs", "by_detection")
# 已知的類別鍵別名（LLM 偶爾會用英文或簡稱代替規定的中文鍵名）
_CATEGORY_KEY_ALIASES = {
    "other": "其他",
    "其它": "其他",
    "misc": "其他",
}


def _normalize_scene_json(obj: dict[str, Any]) -> dict[str, Any]:
    """修正 LLM 輸出中常見的結構偏差：
    1. next_action/answer/regulation_refs/by_detection 誤巢狀塞進 categories → 搬回頂層。
    2. categories 鍵名使用英文/別名（如 "other"）→ 正規化為規定的中文鍵名（"其他"）。
    """
    categories = obj.get("categories")
    if not isinstance(categories, dict):
        return obj

    for key in _TOP_LEVEL_KEYS:
        if key in categories and key not in obj:
            obj[key] = categories.pop(key)
        elif key in categories:
            categories.pop(key)

    for alias, canonical in _CATEGORY_KEY_ALIASES.items():
        if alias in categories:
            existing = categories.get(canonical) or []
            categories[canonical] = existing + categories.pop(alias)

    # 每個類別的值規定必須是陣列；LLM 有時會誤把 item 編號當成巢狀 key（例如
    # "其他":{"環境保護-3":[{...}]}），這裡攤平回規定的陣列形狀，並把遺失的
    # item 欄位補回去。
    for cat_name, value in list(categories.items()):
        if isinstance(value, list):
            continue
        flattened: list = []
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                entries = sub_val if isinstance(sub_val, list) else [sub_val]
                for entry in entries:
                    if isinstance(entry, dict) and not str(entry.get("item") or "").strip():
                        entry = {**entry, "item": sub_key}
                    flattened.append(entry)
        elif value:
            flattened.append(value)
        categories[cat_name] = flattened

    return obj


def _drop_unknown_items(obj: dict[str, Any]) -> dict[str, Any]:
    """凡 item 代碼不在官方稽核表清單（audit_checklist.ALL_KNOWN_CODES）內、
    或代碼與其所屬 categories key 不符（如把「其他不符-1」塞進「墜落防止」）的條目一律捨棄。

    這是「嚴格依據 05招標規範附件.pdf」的最後一道防線：即使 prompt 沒被模型完全遵守、
    生出清單以外的代碼、或把真實代碼套錯類別，也不會流到使用者看到的結果裡。
    """
    categories = obj.get("categories")
    if not isinstance(categories, dict):
        return obj
    for cat_name, items in list(categories.items()):
        if not isinstance(items, list):
            continue
        kept = []
        for it in items:
            if not isinstance(it, dict):
                continue
            code = str(it.get("item") or "").strip()
            expected_cat = CODE_TO_EXPECTED_CATEGORY.get(code)
            if expected_cat is None:
                print(f"[safety_audit] 捨棄非官方項次代碼: {cat_name}/{code!r}")
            elif expected_cat != cat_name:
                print(f"[safety_audit] 捨棄類別錯置的項次: {code!r} 應屬「{expected_cat}」卻出現在「{cat_name}」")
            else:
                # marker/deadline 一律以官方清單為準覆蓋，不採用模型自己填的值
                # （第1~6類稍後 _complete_checklist_1_to_6 還會再覆蓋一次，這裡主要是為了
                # 讓「其他」類別的項目也套用官方 marker，不受模型自由發揮）。
                marker, deadline = official_marker_and_deadline(code)
                it["marker"] = marker
                if str(it.get("result") or "").strip() == "○":
                    it["deadline"] = ""
                else:
                    it["deadline"] = deadline
                kept.append(it)
        categories[cat_name] = kept
    return obj


def _complete_checklist_1_to_6(obj: dict[str, Any]) -> dict[str, Any]:
    """把「墜落防止~交通維持」這6個類別補成完整的54條打勾清單：
    1. 每一條的 marker/deadline 一律以官方清單為準覆蓋（這是 PDF 固定的常數，不該由模型決定，
       模型填什麼都會在這裡被改成正確值）。
    2. 模型漏掉沒列出的條目，補一筆 result="○"（沒被回報異常，視為未發現缺失）的預設項，
       確保最終呈現給使用者的一定是逐條列滿的完整清單，而不是模型自己選擇性列出的殘缺清單。
    """
    categories = obj.setdefault("categories", {})
    if not isinstance(categories, dict):
        categories = {}
        obj["categories"] = categories

    for cat_name, checklist_items in CHECKLIST_1_TO_6.items():
        existing = categories.get(cat_name)
        by_code: dict[str, dict] = {}
        if isinstance(existing, list):
            for it in existing:
                if isinstance(it, dict):
                    by_code[str(it.get("item") or "").strip()] = it

        rebuilt = []
        for ci in checklist_items:
            it = by_code.get(ci.item)
            if it is None:
                it = {"item": ci.item, "result": "○"}
            it["marker"] = ci.marker
            it["deadline"] = _DEADLINE_FOR_MARKER.get(ci.marker, "五日內改善") if str(it.get("result") or "○") != "○" else it.get("deadline", "")
            if str(it.get("result") or "○") == "○":
                # 符合規定的項目不需要期限/描述，保持精簡
                it["deadline"] = ""
                it.setdefault("description", "")
                it.setdefault("location", "")
            rebuilt.append(it)
        categories[cat_name] = rebuilt

    return obj


_DEADLINE_FOR_MARKER = {"☆": "立即停工", "※": "即日改善"}
_MARKER_SEVERITY = {"☆": 2, "※": 1}
_HEDGING_WORDS = ("疑似", "無法確認", "看不清楚", "可能", "不確定", "似乎", "難以判斷", "無法確定")


def _downgrade_uncertain_results(obj: dict[str, Any]) -> dict[str, Any]:
    """description 本身若承認不確定（疑似/無法確認/看不清楚等用語），卻仍判 result=×（缺失確定成立），
    代表證據不足以支撐「確定缺失」的結論（曾實測觀察到模型一邊寫「雖無法確認高空作業設備
    具體位置」一邊仍判×），這裡自動降級為△（應請改善／待確認），避免無憑無據就認定違規。

    marker（☆/※/空字串）現在一律由官方清單的固定屬性決定（見 _complete_checklist_1_to_6／
    official_marker_and_deadline），不再是模型可自由調整的「確定性」表達方式，
    所以這裡改成調整 result，而不是像先前那樣調整 marker。
    """
    categories = obj.get("categories")
    if not isinstance(categories, dict):
        return obj
    for items in categories.values():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or str(it.get("result") or "").strip() != "×":
                continue
            desc = str(it.get("description") or "")
            if any(w in desc for w in _HEDGING_WORDS):
                it["result"] = "△"
    return obj


_CHECKLIST_CODE_PREFIX_RE = re.compile(r"^([0-9]+\.[0-9]+|環保-[0-9]+|其他不符-[0-9]+)\b")


def _looks_like_checklist_code(ref: str) -> bool:
    """判斷一段 regulation_refs 文字是不是把本系統自己的稽核表項次代碼
    （如「1.02」「環保-3」）誤當成外部法規來源填進來。"""
    m = _CHECKLIST_CODE_PREFIX_RE.match(ref.strip())
    return bool(m and m.group(1) in ALL_KNOWN_CODES)


def _ensure_required_fields(obj: dict[str, Any]) -> dict[str, Any]:
    """確保 next_action/answer/regulation_refs 一定有內容。
    LLM 偶爾會漏填這幾個欄位（非結構錯位，而是內容缺漏），導致 LINE 回覆
    或 Flex 卡片上該區塊整個空白；這裡用既有資料（categories/summary）
    產生一個合理的 fallback，而不是留空。
    """
    categories = obj.get("categories") if isinstance(obj.get("categories"), dict) else {}
    all_items = [
        it for items in categories.values() if isinstance(items, list)
        for it in items if isinstance(it, dict)
    ]
    # 只挑「真的有缺失」的項目來當 next_action fallback 的依據；現在 categories 會把
    # 1~6類全部54條（含○符合規定）都列出，若不篩掉○，max() 在同分時會選到清單裡
    # 排序最前面的○項目（如1.01），誤把「符合規定」當成「最嚴重缺失」推薦改善。
    defect_items = [it for it in all_items if str(it.get("result") or "").strip() in ("×", "△")]

    if not str(obj.get("next_action") or "").strip():
        if defect_items:
            worst = max(defect_items, key=lambda it: _MARKER_SEVERITY.get(str(it.get("marker") or ""), 0))
            obj["next_action"] = (
                f"依現場缺失（{worst.get('item', '')}，{worst.get('deadline') or '五日內改善'}）安排改善，"
                "並落實工地安全巡檢與人員教育訓練。"
            )
        else:
            obj["next_action"] = "現場未發現明確缺失，建議持續落實例行工地安全巡檢。"

    if not str(obj.get("answer") or "").strip():
        obj["answer"] = str(obj.get("summary") or "").strip() or "本次分析未提供詳細說明。"

    # regulation_refs 只能是真正的外部法規；曾實測觀察到模型把自己的稽核表項次代碼
    # （如「環保-3」「10.02」）當成法規來源填進來，這裡濾掉開頭是已知項次代碼的條目。
    refs = obj.get("regulation_refs")
    if isinstance(refs, list):
        obj["regulation_refs"] = [
            r for r in refs
            if not (isinstance(r, str) and _looks_like_checklist_code(r))
        ]

    if not obj.get("regulation_refs"):
        obj["regulation_refs"] = ["無明確對應法規，建議由專業技師依現場實際狀況評估"]

    return obj


def extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    start = raw.find("{")
    if start == -1:
        return {"raw_text": raw}
    end = raw.rfind("}")
    if end > start:
        try:
            return _postprocess_llm_json(json.loads(raw[start : end + 1]))
        except json.JSONDecodeError:
            pass
    # 輸出被截斷（JSON 未閉合）時的容錯：嘗試補上閉合括號再解析
    salvaged = _salvage_truncated_json(raw[start:])
    if salvaged is not None:
        return _postprocess_llm_json(salvaged)
    return {"raw_text": raw}


def _postprocess_llm_json(obj: dict[str, Any]) -> dict[str, Any]:
    """LLM 原始 JSON → 可信賴輸出的完整後處理管線，依序：
    1. 結構修正（巢狀錯位、類別鍵別名、非陣列攤平）
    2. 不確定用語 → result 降級（×→△）
    3. 捨棄非官方代碼／類別錯置的項目，並套用官方 marker/deadline
    4. 補齊第1~6類54條的完整打勾清單
    5. 補齊 next_action/answer/regulation_refs 等必填欄位
    """
    obj = _normalize_scene_json(obj)
    obj = _downgrade_uncertain_results(obj)
    obj = _drop_unknown_items(obj)
    obj = _complete_checklist_1_to_6(obj)
    obj = _ensure_required_fields(obj)
    return obj


def _salvage_truncated_json(snippet: str) -> dict[str, Any] | None:
    """嘗試修復被截斷的 JSON：移除尾端不完整片段並補齊括號。

    效能筆記：舊版對每一個字元位置都重新 chunk.count("{"/"}") 一次，
    是 O(n^2)（截斷輸出的字元數 n）。提高 max_tokens 後截斷片段可能更長，
    這裡改用一次性前綴和讓每個候選切點的括號計數變成 O(1)，並只在
    結構安全的邊界（","/"}"/"]"）嘗試切割，同時限制最多嘗試次數。
    """
    n = len(snippet)
    if n == 0:
        return None

    brace_prefix = [0] * (n + 1)
    brack_prefix = [0] * (n + 1)
    for idx, ch in enumerate(snippet):
        brace_prefix[idx + 1] = brace_prefix[idx] + (1 if ch == "{" else -1 if ch == "}" else 0)
        brack_prefix[idx + 1] = brack_prefix[idx] + (1 if ch == "[" else -1 if ch == "]" else 0)

    max_attempts = 500
    attempts = 0
    for i in range(n - 1, -1, -1):
        if snippet[i] not in ",}]":
            continue
        attempts += 1
        if attempts > max_attempts:
            break
        opens_br = brace_prefix[i + 1]
        opens_sq = brack_prefix[i + 1]
        if opens_br < 0 or opens_sq < 0:
            continue
        chunk = snippet[: i + 1].rstrip().rstrip(",")
        candidate = chunk + ("]" * opens_sq) + ("}" * opens_br)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and obj.get("summary"):
                return obj
        except json.JSONDecodeError:
            continue
    return None

def get_llm_client_and_config() -> tuple[OpenAI | None, str, bool]:
    load_dotenv(override=True)
    api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("LLM_MODEL") or "gpt-4o").strip()
    env_vision = os.getenv("LLM_SUPPORTS_VISION", "").strip().lower()
    if env_vision in ("true", "1", "yes"):
        supports_vision = True
    elif env_vision in ("false", "0", "no"):
        supports_vision = False
    else:
        supports_vision = any(k in model.lower() for k in ["gpt-4", "o1", "gemini"])

    if not api_key:
        return None, model, False

    client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None)
    return client, model, supports_vision

def _is_ollama_backend(base_url: str | None) -> bool:
    return bool(base_url) and ("11434" in base_url or "ollama" in base_url.lower())


def _ollama_native_chat(
    base_url: str,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    num_ctx: int,
    use_json_format: bool,
    disable_thinking: bool,
) -> str:
    """繞過 Ollama 這個版本 OpenAI 相容端點（/v1/chat/completions）的一個實測缺陷：
    無論透過 extra_body 傳多大的 num_ctx，該端點回報的 total_tokens 都被錨死在 4096，
    導致視覺模型（圖片編碼本身就可能吃掉 5000+ tokens）+ 本系統的完整54條清單 prompt
    組合起來時，輸出常常被腰斬成空字串（finish_reason="length" 但 content 是空的）。
    改打 Ollama 原生 /api/chat 端點則能正確套用 num_ctx／think，故直連此端點取代
    openai client 呼叫，避免每次分析都因為這個相容層問題而分析失敗。
    """
    import requests

    api_base = base_url.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[: -len("/v1")]

    native_messages: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            native_messages.append({"role": m["role"], "content": content})
            continue
        texts: list[str] = []
        images: list[str] = []
        for part in content or []:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:") and "," in url:
                    images.append(url.split(",", 1)[1])
        entry: dict[str, Any] = {"role": m["role"], "content": "\n".join(texts)}
        if images:
            entry["images"] = images
        native_messages.append(entry)

    payload: dict[str, Any] = {
        "model": model,
        "messages": native_messages,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": max_tokens},
    }
    if use_json_format:
        payload["format"] = "json"
    if disable_thinking:
        payload["think"] = False

    resp = requests.post(f"{api_base}/api/chat", json=payload, timeout=240)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message") or {}).get("content") or ""


def ask_llm_scene_analysis(
    client: OpenAI,
    model: str,
    image_path: Path,
    prompt: str,
    detections: list[dict],
    supports_vision: bool,
    use_yolo: bool,
    annotated_path: Path | None = None,
) -> dict[str, Any]:
    # 檢索法規知識庫（query 含偵測類別）
    reg_dir = Path(__file__).parent / "regulations"
    retrieval_query = build_regulation_retrieval_query(prompt, detections if use_yolo else [])
    snippets = retrieve_regulations(retrieval_query, reg_dir)
    context_text = format_snippets(snippets)

    detection_block = format_detections_for_prompt(detections) if use_yolo else format_detections_for_prompt([])

    full_prompt = (
        f"{prompt}\n\n"
        f"{detection_block}\n\n"
        f"[官方稽核項次清單（第1~6類，逐條核對，只回報有具體證據的缺失）]:\n{render_checklist_1_to_6()}\n\n"
        f"[一般風險／環境保護參考清單（「其他」類別缺失須從此挑選真實代碼）]:\n{render_other_reference_items()}\n\n"
        f"[參考法規]:\n{context_text}"
    )

    try:
        messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
        
        if supports_vision:
            user_parts: list[dict[str, Any]] = [{"type": "text", "text": full_prompt}]
            try:
                with open(image_path, "rb") as f:
                    orig_url = build_data_url(f.read())
                user_parts.append({"type": "text", "text": "【原圖】完整現場，供補充未框註之細節與整體環境。"})
                user_parts.append({"type": "image_url", "image_url": {"url": orig_url}})

                # 有偵測時另傳標註圖，讓框線與 detection_index 對照
                show_annotated = (
                    use_yolo
                    and bool(detections)
                    and annotated_path is not None
                    and annotated_path.exists()
                    and annotated_path.resolve() != image_path.resolve()
                )
                if show_annotated:
                    try:
                        with open(annotated_path, "rb") as f:
                            ann_url = build_data_url(f.read())
                        user_parts.append({"type": "text", "text": "【YOLO 標註圖】方框為模型偵測位置；類別與座標以文字清單為準。"})
                        user_parts.append({"type": "image_url", "image_url": {"url": ann_url}})
                    except OSError as _ann_err:
                        print(f"[LLM] 標註圖讀取失敗，略過：{_ann_err}")

                messages.append({"role": "user", "content": user_parts})
            except OSError as _img_err:
                print(f"[LLM] 原圖讀取失敗，退回純文字模式：{_img_err}")
                messages.append({"role": "user", "content": full_prompt})
        else:
            messages.append({"role": "user", "content": full_prompt})

        # OpenAI、Ollama 等相容端點皆支援 json_object；預設一律啟用以確保輸出可解析
        use_json_format = os.getenv("LLM_FORCE_JSON", "1").strip() != "0"
        create_kwargs: dict[str, Any] = {"model": model, "messages": messages}
        # 限制輸出長度上限，避免模型話太多被中途截斷導致 JSON 不完整、解析失敗
        # 完整 schema 現在要求逐條列出54條官方項次（categories）+ next_action/answer/
        # regulation_refs/by_detection，實測輸出常需要 1500~2500 completion tokens，
        # 3072 邊界太緊，提高到 4096 留一點餘裕。
        create_kwargs["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        if use_json_format:
            create_kwargs["response_format"] = {"type": "json_object"}
        # Ollama 預設 context 僅 4096。實測發現：視覺模型的圖片編碼本身就可能吃掉
        # 5000~7000 tokens（且不一定會反映在 OpenAI 相容層回報的 prompt_tokens 裡），
        # 加上本系統的 54 條官方清單 prompt（近2000 tokens）與逐條 JSON 輸出，
        # 8192 常常不夠、導致輸出被腰斬變成空字串（finish_reason="length" 但 content
        # 是空的）。調高到 16384 給圖片＋長 prompt＋完整輸出足夠空間。
        num_ctx = int(os.getenv("LLM_NUM_CTX", "16384"))
        extra_body: dict[str, Any] = {}
        if num_ctx > 0:
            # Ollama 的 OpenAI 相容端點需把 num_ctx 包在 options 內才會生效
            extra_body["options"] = {"num_ctx": num_ctx}
        # 部分 Ollama 模型（如 gemma4）預設會先輸出「思考」內容再給答案，
        # 但思考內容算在 max_tokens 額度內、且不會出現在 content 欄位——
        # 實測 max_tokens=3072 時思考直接吃光額度，content 變成空字串、
        # finish_reason="length"，導致 extract_json 完全解析不到東西。
        # 用 think:false 關閉思考模式，讓輸出額度全部留給實際要的 JSON。
        disable_thinking = os.getenv("LLM_DISABLE_THINKING", "1").strip() != "0"
        if disable_thinking:
            extra_body["think"] = False
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        base_url = os.getenv("LLM_BASE_URL") or ""
        if _is_ollama_backend(base_url):
            content = _ollama_native_chat(
                base_url, messages, model,
                max_tokens=create_kwargs["max_tokens"],
                num_ctx=num_ctx,
                use_json_format=use_json_format,
                disable_thinking=disable_thinking,
            )
        else:
            response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content
        return extract_json(content)
    except Exception as e:
        print(f"[LLM] 呼叫失敗 ({type(e).__name__}): {e}")
        return {"summary": f"LLM 呼叫失敗: {str(e)}", "risks": ["API 連線異常"]}


def resolve_yolo_model_path(yolo_model_path: str | Path) -> str:
    """
    相對路徑以本檔所在專案根為準。
    找不到指定檔時，再嘗試專案根目錄 best.pt；仍沒有則使用 yolov8n.pt。
    """
    base = Path(__file__).resolve().parent
    p = Path(yolo_model_path)
    if not p.is_absolute():
        p = base / p
    if p.exists():
        return str(p)
    root_best = base / "best.pt"
    if root_best.exists():
        return str(root_best)
    print(f"警告: 找不到模型 {yolo_model_path}，嘗試使用 yolov8n.pt")
    return "yolov8n.pt"


def _should_slice(image_path: Path) -> bool:
    """依設定與圖片長邊判斷這張圖要不要走切片偵測。

    只需要圖片尺寸，用 PIL 讀 header 即可（不解碼像素資料），
    避免對大圖（切片偵測正是為大圖設計的）多花一次完整 cv2.imread 解碼的成本，
    這張圖後面 _run_full_detection/run_sliced_detection 跟 enrich_detections_spatial
    還會各自再讀一次，這裡沒必要跟著解碼一次全尺寸像素。
    """
    if not YOLO_USE_SLICING:
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception:
        return False
    return max(h, w) > YOLO_SLICE_TRIGGER_SIDE


def _draw_detections(image_path: Path, detections: list[dict], annotated_file: Path) -> None:
    """把偵測框畫到原圖上並存成 annotated_file。切片模式沒有 YOLO 內建的 r.plot()，
    所以自己用 cv2 畫框＋標籤，維持與單次推論一致的標註圖輸出。"""
    im = cv2.imread(str(image_path))
    if im is None:
        shutil.copy(image_path, annotated_file)
        return
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        label = f"{d.get('yolo_class', '')} {float(d.get('confidence', 0)):.2f}"
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(im, label, (x1 + 2, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(annotated_file), im):
        print(f"[YOLO] 切片標註圖寫入失敗：{annotated_file}")
        shutil.copy(image_path, annotated_file)


def run_sliced_detection(
    image_path: Path, yolo_model_path: str, min_conf: float, annotated_file: Path
) -> list[dict]:
    """用 SAHI 對大圖做切片偵測：切成有重疊的小塊、逐塊跑 YOLO、再用 NMS 合併，
    回傳與單次推論相同格式的 parsed_items（yolo_class / confidence / box）。
    SAHI 依 slice 尺寸與圖片大小自動決定要切幾塊，不需手動指定塊數。"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # 切片放大雜物易誤判，故用專屬（通常較嚴）的 YOLO_SLICE_MIN_CONF，
    # 但不低於整張推論的 min_conf，避免使用者把整張門檻調高後切片反而更寬鬆。
    slice_conf = max(min_conf, YOLO_SLICE_MIN_CONF)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=yolo_model_path,
        confidence_threshold=slice_conf,
    )
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=YOLO_SLICE_SIZE,
        slice_width=YOLO_SLICE_SIZE,
        overlap_height_ratio=YOLO_SLICE_OVERLAP,
        overlap_width_ratio=YOLO_SLICE_OVERLAP,
        verbose=0,
    )
    parsed_items: list[dict] = []
    for obj in result.object_prediction_list:
        box = obj.bbox.to_xyxy()  # [x1, y1, x2, y2]
        parsed_items.append({
            "yolo_class": obj.category.name,
            "confidence": float(obj.score.value),
            "box": [float(v) for v in box],
        })
    _draw_detections(image_path, parsed_items, annotated_file)
    return parsed_items


def _run_full_detection(
    image_path: Path, yolo_model_path: str, min_conf: float, annotated_file: Path
) -> list[dict]:
    """單次整張推論（原本的做法）：直接對整張圖跑 YOLO，並用 r.plot() 存標註圖。"""
    parsed_items: list[dict] = []
    model = YOLO(yolo_model_path)
    results = model.predict(source=image_path, conf=min_conf, save=False)
    for r in results:
        im_array = r.plot()
        if not cv2.imwrite(str(annotated_file), im_array):
            print(f"[YOLO] 標註圖寫入失敗：{annotated_file}")
        for box in r.boxes:
            try:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                parsed_items.append({
                    "yolo_class": name,
                    "confidence": float(box.conf[0]),
                    "box": box.xyxy[0].tolist(),
                })
            except (IndexError, RuntimeError) as _box_err:
                print(f"[YOLO] box 解析失敗，跳過：{_box_err}")
    return parsed_items


def analyze_image(
    image_path: str | Path,
    yolo_model_path: str | Path = "best.pt",
    llm_model: str = "gpt-4o",
    output_dir: str | Path = "outputs",
    min_conf: float = 0.25,
    class_keywords: list[str] | None = None,
    prompt: str = "請分析這張施工現場照片的職安合規性。",
    use_yolo: bool = True,
) -> dict[str, Any]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 建立本次任務的獨立資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"audit_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    
    annotated_file = session_dir / "annotated.jpg"
    report_file = session_dir / "report.json"

    parsed_items = []
    
    # 1. YOLO 偵測
    if use_yolo:
        try:
            yolo_model_path = resolve_yolo_model_path(yolo_model_path)

            # 大圖走 SAHI 切片偵測（提升遠處小物件召回率）；小圖維持單次整張推論。
            if _should_slice(image_path):
                try:
                    print(f"[YOLO] 大圖啟用切片偵測（slice={YOLO_SLICE_SIZE}, overlap={YOLO_SLICE_OVERLAP}）")
                    parsed_items = run_sliced_detection(
                        image_path, yolo_model_path, min_conf, annotated_file
                    )
                except Exception as slice_err:
                    # 切片失敗（如 sahi 未安裝）時退回單次整張推論，確保流程不中斷
                    print(f"[YOLO] 切片偵測失敗，退回單次整張推論：{slice_err}")
                    parsed_items = _run_full_detection(
                        image_path, yolo_model_path, min_conf, annotated_file
                    )
            else:
                parsed_items = _run_full_detection(
                    image_path, yolo_model_path, min_conf, annotated_file
                )

            # 如果沒畫到圖（沒偵測到任何東西），也複製一份原圖確保檔案存在
            if not annotated_file.exists():
                shutil.copy(image_path, annotated_file)

        except Exception as e:
            print(f"YOLO 偵測錯誤: {e}")
            shutil.copy(image_path, annotated_file)
    else:
        shutil.copy(image_path, annotated_file)

    # 補上相對座標與區域（供 LLM 與報告 JSON 使用）
    _im = cv2.imread(str(image_path))
    if _im is not None and parsed_items:
        _h, _w = _im.shape[:2]
        enrich_detections_spatial(parsed_items, _w, _h)

    # 2. LLM 分析
    client, effective_model, supports_vision = get_llm_client_and_config()
    
    if client:
        scene_analysis = ask_llm_scene_analysis(
            client=client,
            model=effective_model,
            image_path=image_path,
            prompt=prompt,
            detections=parsed_items,
            supports_vision=supports_vision,
            use_yolo=use_yolo,
            annotated_path=annotated_file if use_yolo else None,
        )
    else:
        scene_analysis = {
            "summary": "未設定 LLM API Key，僅完成物件偵測。",
            "detected_items": [item["yolo_class"] for item in parsed_items],
            "risks": ["無法進行深度風險評估"],
            "next_action": "請檢查 .env 檔案中的 API 設定。",
            "answer": "僅顯示偵測到的物件，無法提供職安建議。"
        }

    # 3. 整理結果
    result = {
        "timestamp": timestamp,
        "input_image": str(image_path.absolute()),
        "annotated_image": str(annotated_file.absolute()),
        "detections": parsed_items,
        "scene_analysis": scene_analysis,
    }

    # 儲存 JSON 紀錄
    report_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return result

if __name__ == "__main__":
    # 簡單的命令列測試
    args = parse_args()
    if Path(args.image).exists():
        res = analyze_image(image_path=args.image, yolo_model_path=args.yolo_model)
        print(f"分析完成！結果儲存在: {res['annotated_image']}")
    else:
        print("請提供有效的圖片路徑。")
