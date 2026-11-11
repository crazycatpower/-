# -*- coding: utf-8 -*-
"""
稽核 VLM（Gemma 3n E2B + LoRA）微調與評測的共用程式：模型載入、KV 共用修補、prompt、
YOLO 偵測清單、輸出解析、答案表讀取。finetune_gemma3n_v10.py、test_val57.py、
build_finetune_dataset_v10.py 都從這裡取，訓練與評測才不會各寫一份而走樣。

為什麼需要 patch_kv_sharing（v1~v9 卡關的原因）
------------------------------------------------
Gemma 3n 的第 20~29 層不自己算 key/value，而是共用第 18、19 層算好的（KV sharing）。
transformers 5.0.0.dev0 把這個共用寫在 `past_key_values is not None` 的分支裡，也就是只有
use_cache=True 才會共用。generate() 推論時 use_cache=True，沒問題；但訓練時 TRL 的
SFTTrainer 和 gradient checkpointing 都會把 cache 關掉，第 20~29 層就改用自己那組
從沒被用過的 k_proj/v_proj，等於換了一個模型：

    4-bit 底座、24 張訓練圖的 loss      use_cache=True（推論）   use_cache=False（訓練）
    不掛 adapter                          2.02                     6.25
    掛 v6 adapter                         1.20                     0.04

v6 在「訓練用的那個模型」上把訓練集背到 loss 0.04，換回推論用的模型幾乎沒有效果，
所以 v1~v9 不管怎麼加資料，val57 都停在底座 zero-shot 附近。patch_kv_sharing 讓沒有
cache 的 forward 也照樣共用 KV，訓練和推論才是同一個模型。
"""
import inspect
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "google/gemma-3n-E2B-it"
IMG_DIR = BASE_DIR / "uploads" / "UploadImg"
ANSWER_XLSX = BASE_DIR / "答案.xlsx"
EXT_DIR = BASE_DIR / "新增資料集"
EXT_XLSX = EXT_DIR / "外部來源_答案.xlsx"
VAL_JSONL = BASE_DIR / "finetune_data" / "val.jsonl"

# 與 v5~v9 的訓練、評測逐字相同，val57 的數字才能跟歷史版本比
SYSTEM_PROMPT = (
    "你是專業的工地職安稽核員。分析照片後只輸出 JSON："
    '{"compliant": true/false, "findings": [{"item":"項目名稱","result":"缺失描述"}]}'
    "，符合規定時 findings 為空陣列。"
)
PROMPT_PREFIX = "請分析這張施工現場照片的職安合規性。"
ITEM_HINT = "本張照片的稽核項目：{}"
COMPLIANT = "符合規定"


# ── 模型 ────────────────────────────────────────────────────────────────

def resolve_model_path():
    """本機已有快取就只讀本機（避免新版 transformers 對 gated repo 的多餘網路請求）。"""
    snaps = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-3n-E2B-it" / "snapshots"
    local = next(snaps.glob("*"), None) if snaps.exists() else None
    if local:
        return str(local), {"local_files_only": True}
    return MODEL_ID, {"token": os.getenv("HF_TOKEN") or None}


class _CpuEmbedding(torch.nn.Module):
    """per-layer embedding（約 20 億參數、bf16 4GB）留在 CPU：每個 token 只查表一次，幾乎不影響速度。
    內層模組放在 list 裡不註冊成子模組，model.to(gpu) 才不會把它搬上去。"""

    def __init__(self, inner, out_device):
        super().__init__()
        self._inner = [inner]
        self.out_device = out_device

    def forward(self, input_ids):
        return self._inner[0](input_ids.to("cpu")).to(self.out_device)


class _AudioDisabled(torch.nn.Module):
    """音訊編碼器（約 7 億參數）用不到，拿掉省 VRAM。"""

    def forward(self, *args, **kwargs):
        raise RuntimeError("這個流程不處理音訊")


def patch_kv_sharing(model):
    """讓 use_cache=False 的 forward 也做 KV 共用（說明見檔頭）。上游若已修掉就不動。"""
    from transformers.models.gemma3n import modeling_gemma3n as mg

    cls = mg.Gemma3nTextAttention
    if not getattr(cls, "_kv_share_patched", False):
        if "self.is_kv_shared_layer and past_key_values is not None" in inspect.getsource(cls.forward):
            original = cls.forward

            def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                        past_key_values=None, cache_position=None, **kwargs):
                store = getattr(self, "_shared_kv", None)
                if past_key_values is not None or store is None:
                    return original(self, hidden_states, position_embeddings, attention_mask,
                                    past_key_values, cache_position, **kwargs)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.config.head_dim)
                cos, sin = position_embeddings
                query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
                query = mg.apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)

                if self.is_kv_shared_layer:
                    key, value = store[self.kv_shared_layer_index]
                else:
                    key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
                    key = mg.apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                    value = self.v_norm(self.v_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    if self.store_full_length_kv:
                        store[self.layer_idx] = (key, value)

                attention = mg.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention = mg.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                out, weights = attention(
                    self, query, key, value, attention_mask,
                    dropout=self.attention_dropout if self.training else 0.0,
                    scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
                )
                out = out.reshape(*input_shape, -1).contiguous()
                return self.o_proj(out), weights

            cls.forward = forward
        cls._kv_share_patched = True

    store = {}
    for m in model.modules():
        if isinstance(m, cls):
            m._shared_kv = store


def _patch_sdpa_gqa():
    """Windows 版 torch 沒有 Flash Attention 時，SDPA 的 GQA 路徑會退回 math kernel 建出整個注意力矩陣；
    改成先展開 KV head，走 memory-efficient kernel（與 llm_server.py 相同）。"""
    probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    if probe is not None and probe():
        return
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def load_model(adapter=None, merge=False, device="cuda:0"):
    """回傳 (model, processor)。bf16、不量化；PLE 留在 CPU、拿掉音訊編碼器，GPU 約佔 5GB。

    adapter: LoRA 目錄；None 表示只載底座。merge=True 會把 LoRA 併進權重（llm_server.py 的部署方式）。
    """
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    path, kwargs = resolve_model_path()
    processor = AutoProcessor.from_pretrained(path, **kwargs)
    # 先整個載到 CPU 再決定誰上 GPU：混合 device_map 會把 CPU 模組當成「執行時搬上 GPU」，反而佔滿 VRAM
    model = Gemma3nForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cpu", **kwargs)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        if merge:
            model = model.merge_and_unload()
    place_on_gpu(model, device)
    model.eval()
    return model, processor


def place_on_gpu(model, device="cuda:0"):
    """把模型搬上 GPU，但 PLE 留 CPU、音訊編碼器換成空殼；並套用 KV 共用與 SDPA 修補。"""
    device = torch.device(device)
    for name, module in list(model.named_modules()):
        if "." not in name:
            continue
        parent, leaf = name.rsplit(".", 1)
        if leaf == "audio_tower":
            setattr(model.get_submodule(parent), leaf, _AudioDisabled())
        elif leaf == "embed_tokens_per_layer" and not isinstance(module, _CpuEmbedding):
            setattr(model.get_submodule(parent), leaf, _CpuEmbedding(module, device))
    model.to(device)
    patch_kv_sharing(model)
    _patch_sdpa_gqa()
    return model


def gemma_of(model):
    """PeftModel 或裸模型 → Gemma3nForConditionalGeneration。"""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


# ── 輸入 ────────────────────────────────────────────────────────────────

def load_image(path):
    """依 EXIF 轉正：YOLO 讀圖時會轉正，VLM 這邊不轉的話手機直拍照會躺著進模型，跟 YOLO 座標對不上。"""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def user_prompt(detection_block, item=None):
    text = f"{PROMPT_PREFIX}\n\n{detection_block}"
    if item:
        text += "\n\n" + ITEM_HINT.format(item)
    return text


def encode(processor, image, prompt_text, response=None):
    """組出模型輸入。給 response 時回傳 (batch, labels)，labels 只在答案上有值（其餘 -100）；
    不給時回傳 (batch, None)，結尾停在 <start_of_turn>model 等模型作答。"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=response is None)
    batch = processor(text=[text], images=[[image]], return_tensors="pt")
    if response is None:
        return batch, None

    marker = processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
    ids = batch["input_ids"][0].tolist()
    start = next(i for i in range(len(ids) - len(marker), -1, -1) if ids[i:i + len(marker)] == marker) + len(marker)
    labels = batch["input_ids"].clone()
    labels[0, :start] = -100
    return batch, labels


def decision_token_ids(tokenizer):
    """答案裡 true / false 那個 token 的 id，以及它是答案的第幾個 token。"""
    a = tokenizer.encode('{"compliant": true, "findings": []}', add_special_tokens=False)
    b = tokenizer.encode('{"compliant": false, "findings": [', add_special_tokens=False)
    pos = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    return pos, a[pos], b[pos]


def to_device(model, batch):
    head = gemma_of(model).lm_head.weight
    out = {k: v.to(head.device) for k, v in batch.items()}
    out["pixel_values"] = out["pixel_values"].to(head.dtype)
    return out


def answer_logits(model, batch, labels):
    """只在答案的位置過 lm_head，回傳 (logits[n, vocab], targets[n])。
    整段序列都過 lm_head 要 seq_len × 262400 的 logits，長 prompt 會吃掉好幾 GB。"""
    gemma = gemma_of(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = gemma.model(**to_device(model, batch), use_cache=False).last_hidden_state[0]
        targets = labels[0, 1:].to(hidden.device)
        keep = targets != -100
        logits = gemma.lm_head(hidden[:-1][keep])
    logits = logits.float()
    cap = gemma.config.get_text_config().final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits, targets[keep]


@torch.no_grad()
def generate_answer(model, processor, image, prompt_text, max_new_tokens=512):
    """greedy 生成答案。回傳 (文字, p_violation)：p_violation 是模型寫到 true/false 那一格時選 false 的機率，
    沒寫到那一格（輸出壞掉）則為 None。"""
    from transformers import LogitsProcessor, LogitsProcessorList

    _, t_id, f_id = decision_token_ids(processor.tokenizer)

    class Probe(LogitsProcessor):
        def __init__(self):
            self.steps = []

        def __call__(self, input_ids, scores):
            self.steps.append(scores[0, [t_id, f_id]].float().cpu())
            return scores

    probe = Probe()
    batch, _ = encode(processor, image, prompt_text)
    inputs = to_device(model, batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             logits_processor=LogitsProcessorList([probe]))
    new_tokens = out[0][inputs["input_ids"].shape[1]:].tolist()
    step = next((i for i, t in enumerate(new_tokens) if t in (t_id, f_id)), None)
    p_violation = None if step is None else torch.softmax(probe.steps[step], 0)[1].item()
    return processor.decode(new_tokens, skip_special_tokens=True), p_violation


# ── YOLO 偵測清單 ─────────────────────────────────────────────────────────

_REGIONS = (("上左", "上中", "上右"), ("中左", "中央", "中右"), ("下左", "下中", "下右"))


class Yolo:
    """輸出文字與 benchmark.run_yolo 逐字相同，但模型只載一次，而且路徑含中文也讀得到圖片尺寸
    （run_yolo 用的 cv2.imread 讀不了中文路徑）。"""

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(str(BASE_DIR / "best.pt"))

    def block(self, image_path):
        lines = []
        for r in self.model.predict(source=str(image_path), conf=0.25, save=False, verbose=False):
            h, w = r.orig_shape
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                region = _REGIONS[min(2, int(cy * 3))][min(2, int(cx * 3))]
                lines.append(
                    f"  #{len(lines) + 1} 類別={self.model.names[int(box.cls[0])]} "
                    f"信心={round(float(box.conf[0]), 3):.2f} 區域={region} "
                    f"norm_xyxy=({round(x1 / w, 4)},{round(y1 / h, 4)})-({round(x2 / w, 4)},{round(y2 / h, 4)})"
                )
        if not lines:
            return "（YOLO 未偵測到任何物件；請依原圖做一般職安觀察。）"
        return "\n".join(["[YOLO 偵測清單 — 請逐項對照圖片與法規]"] + lines)


# ── 輸出解析、答案表 ─────────────────────────────────────────────────────

def predict_violation(text):
    """模型輸出 → 是否判定有缺失（findings 非空）；解析失敗回 None。與 v5~v9 的評測相同。"""
    t = text.strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return bool(json.loads(t[s:e + 1]).get("findings", []))
    except Exception:
        return None


def read_answers(xlsx):
    """答案表（沒有標題列：A 圖片路徑、D 稽核項目、E 結果）→ {圖檔名: [(項目, 結果), ...]}。"""
    import openpyxl

    out = {}
    for row in openpyxl.load_workbook(xlsx).worksheets[0].iter_rows(values_only=True):
        if not row[0] or row[4] is None:
            continue
        name = str(row[0]).replace("\\", "/").split("/")[-1]
        out.setdefault(name, []).append((str(row[3] or "").strip(), str(row[4]).strip()))
    return out


def response_for(rows):
    """一張圖的 (項目, 結果) 列 → 訓練／比對用的答案 JSON 字串。"""
    findings = [{"item": item, "result": result} for item, result in rows if result != COMPLIANT]
    return json.dumps({"compliant": not findings, "findings": findings}, ensure_ascii=False)


def item_hint_for(rows):
    """稽核項目提示：去掉項目名稱結尾的冒號；同一張圖多列時去重後用頓號串起來。"""
    items = []
    for item, _ in rows:
        item = item.rstrip("：:").strip() or "未指定"
        if item not in items:
            items.append(item)
    return "、".join(items)
