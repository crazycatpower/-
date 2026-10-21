#!/usr/bin/env python3
"""
Gemma 3n E2B + 稽核 LoRA 推論服務（OpenAI 相容端點）

以 transformers + peft 載入 google/gemma-3n-E2B-it 與 gemma3n-e2b-audit-lora/final，
對外提供 /v1/chat/completions，讓 safety_audit_system.py 只要改 .env 就能切換：

  LLM_BASE_URL=http://127.0.0.1:8001/v1
  LLM_API_KEY=local
  LLM_MODEL=gemma3n-e2b-audit-lora
  LLM_SUPPORTS_VISION=true

啟動（conda safety-bot 環境）：
  pip install -r requirements-lora.txt
  hf auth login                    # base model 為 gated，需先在 HF 網頁同意 Gemma 授權
  python llm_server.py

環境變數（皆選填）：
  LORA_BASE_MODEL   base model（預設 google/gemma-3n-E2B-it）
  LORA_ADAPTER_DIR  adapter 目錄（預設 ./gemma3n-e2b-audit-lora/final）
  LORA_OFFLOAD_PLE  1 | 0（預設 1：per-layer embedding 放 CPU，省下約 4GB VRAM）

記憶體配置（8GB 顯卡）：LoRA 先合併進權重，整個模型以 bf16 放 GPU（約 5.5GB），
但 per-layer embedding（約 4GB）留在 CPU、用不到的音訊編碼器直接移除。
  LORA_HOST / LORA_PORT  綁定位址（預設 127.0.0.1:8001）
  LORA_MODEL_NAME   對外回報的模型名稱（預設 gemma3n-e2b-audit-lora）
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

load_dotenv()

BASE_DIR = Path(__file__).parent
BASE_MODEL = os.getenv("LORA_BASE_MODEL", "google/gemma-3n-E2B-it")
ADAPTER_DIR = Path(os.getenv("LORA_ADAPTER_DIR", str(BASE_DIR / "gemma3n-e2b-audit-lora" / "final")))
OFFLOAD_PLE = os.getenv("LORA_OFFLOAD_PLE", "1").strip() != "0"
HOST = os.getenv("LORA_HOST", "127.0.0.1")
PORT = int(os.getenv("LORA_PORT", "8001"))
MODEL_NAME = os.getenv("LORA_MODEL_NAME", "gemma3n-e2b-audit-lora")
DEFAULT_MAX_TOKENS = 4096


# ── 模型載入 ─────────────────────────────────────────────────────────────

class _CpuEmbedding(torch.nn.Module):
    """把留在 CPU 的 embedding 包起來：輸入搬到 CPU 查表，輸出再搬回 GPU。

    Gemma 3n 的 per-layer embedding（PLE）約 20 億參數、bf16 約 4GB，
    但每個 token 只查表一次，放 CPU 幾乎不影響速度，卻能讓 8GB 顯卡放得下整個模型。
    內層模組刻意不註冊成子模組，之後 model.to(gpu) 才不會把它一起搬上 GPU。
    """

    def __init__(self, inner: torch.nn.Module, out_device: torch.device):
        super().__init__()
        self._inner = [inner]
        self.out_device = out_device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._inner[0](input_ids.to("cpu")).to(self.out_device)


class _AudioDisabled(torch.nn.Module):
    """取代音訊編碼器（約 7 億參數）；本服務只處理文字與圖片。"""

    def forward(self, *args, **kwargs):
        raise RuntimeError("此服務不支援音訊輸入")


def _patch_sdpa_gqa():
    """Windows 版 PyTorch 沒有編譯 Flash Attention 時，關掉 transformers 的 SDPA GQA 路徑。

    transformers 在沒有 attention mask 時會帶 enable_gqa=True 呼叫 SDPA，預期由 Flash
    Attention 處理；但 Windows 版 torch 沒有 Flash，memory-efficient kernel 又不支援
    enable_gqa，只能退回 math kernel，prefill 時建出 heads×L×L 的注意力矩陣——
    稽核 prompt 上萬 token 時要好幾 GB，8GB 顯卡直接 OOM。
    改成先把 KV head 複製展開，就能走 memory-efficient kernel。
    """
    if torch.backends.cuda.is_flash_attention_available():
        return
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
    print("[LoRA] 此 torch 無 Flash Attention，SDPA 改為展開 KV head 以使用 memory-efficient kernel")


def load_model():
    from peft import PeftModel
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU；請在 conda safety-bot 環境（torch cu128）執行")
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"找不到 LoRA adapter：{ADAPTER_DIR}")

    gpu = torch.device("cuda:0")

    # 先整個載到 CPU，手動決定哪些模組上 GPU（transformers 的混合 device_map
    # 會把 CPU 模組當成「存在 CPU、執行時搬上 GPU」，反而佔滿 VRAM）
    print(f"[LoRA] 載入 base model：{BASE_MODEL}")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"[LoRA] 套用並合併 adapter：{ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).merge_and_unload()
    model.eval()

    for name, module in list(model.named_modules()):
        if "." not in name:
            continue
        parent_name, leaf = name.rsplit(".", 1)
        if leaf == "audio_tower":
            setattr(model.get_submodule(parent_name), leaf, _AudioDisabled())
        elif leaf == "embed_tokens_per_layer" and OFFLOAD_PLE:
            setattr(model.get_submodule(parent_name), leaf, _CpuEmbedding(module, gpu))

    model.to(gpu)
    _patch_sdpa_gqa()

    # processor（含 chat template）以訓練時存下的版本為準
    processor = AutoProcessor.from_pretrained(str(ADAPTER_DIR))

    used = torch.cuda.memory_allocated(gpu) / 1024**3
    print(f"[LoRA] 模型就緒，GPU 已用 {used:.2f} GB")
    return model, processor, gpu


# ── 請求轉換 ─────────────────────────────────────────────────────────────

def _load_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        data = base64.b64decode(b64)
    elif url.startswith(("http://", "https://")):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.content
    else:
        raise ValueError("image_url 只支援 data URL 或 http(s) 網址")
    return Image.open(io.BytesIO(data)).convert("RGB")


def to_gemma_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI 格式 → Gemma 3n processor 格式（content 一律轉成 list，圖片轉 PIL）。"""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool":
            raise HTTPException(400, "此服務不支援 tool 訊息")
        content = m.get("content")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            for p in content or []:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    try:
                        parts.append({"type": "image", "image": _load_image(url)})
                    except Exception as e:
                        raise HTTPException(400, f"圖片解析失敗：{e}")

        if role == "system":
            # chat template 只取 system 的第一段文字，先合併成一段
            text = "\n".join(p["text"] for p in parts if p["type"] == "text")
            parts = [{"type": "text", "text": text}]

        # template 要求 user/assistant 交替；相鄰同角色訊息合併
        if out and out[-1]["role"] == role and role != "system":
            out[-1]["content"].extend(parts)
        else:
            out.append({"role": role, "content": parts})
    return out


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_json_fence(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


# ── 推論 ─────────────────────────────────────────────────────────────────

MODEL = PROCESSOR = DEVICE = None
GPU_LOCK = threading.Lock()  # 單張 GPU，一次只跑一個請求


def _prepare(body: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    messages = to_gemma_messages(body.get("messages") or [])
    if not messages:
        raise HTTPException(400, "messages 不可為空")

    inputs = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    temperature = float(body.get("temperature") or 0)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        if body.get("top_p") is not None:
            gen_kwargs["top_p"] = float(body["top_p"])
    return inputs, gen_kwargs


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def generate(body: dict[str, Any]) -> dict[str, Any]:
    json_mode = (body.get("response_format") or {}).get("type") == "json_object"
    with GPU_LOCK:
        inputs, gen_kwargs = _prepare(body)
        prompt_len = inputs["input_ids"].shape[-1]
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = MODEL.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0
    new_tokens = output[0][prompt_len:]
    print(f"[LoRA] prompt={prompt_len} completion={len(new_tokens)} tokens，耗時 {elapsed:.1f}s "
          f"({len(new_tokens) / max(elapsed, 1e-6):.1f} tok/s)")
    text = PROCESSOR.decode(new_tokens, skip_special_tokens=True).strip()
    if json_mode:
        text = _strip_json_fence(text)
    finish = "length" if len(new_tokens) >= gen_kwargs["max_new_tokens"] else "stop"
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_tokens),
            "total_tokens": prompt_len + len(new_tokens),
        },
    }


def generate_stream(body: dict[str, Any]):
    from transformers import TextIteratorStreamer

    cid, created = _completion_id(), int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        data = {
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    with GPU_LOCK:
        inputs, gen_kwargs = _prepare(body)
        streamer = TextIteratorStreamer(PROCESSOR.tokenizer, skip_prompt=True, skip_special_tokens=True)
        errors: list[BaseException] = []

        def run():
            try:
                with torch.inference_mode():
                    MODEL.generate(**inputs, **gen_kwargs, streamer=streamer)
            except BaseException as e:  # 讓串流端知道生成失敗，而不是無聲結束
                errors.append(e)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        yield chunk({"role": "assistant", "content": ""})
        for piece in streamer:
            if piece:
                yield chunk({"content": piece})
        worker.join()
    if errors:
        yield chunk({"content": f"\n[生成失敗：{errors[0]}]"}, "stop")
    else:
        yield chunk({}, "stop")
    yield "data: [DONE]\n\n"


# ── HTTP 端點 ────────────────────────────────────────────────────────────

app = FastAPI(title="Gemma 3n Audit LoRA")


@app.get("/health")
def health():
    return {"status": "ok" if MODEL is not None else "loading", "model": MODEL_NAME}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if MODEL is None:
        raise HTTPException(503, "模型尚未載入完成")
    body = await request.json()
    if body.get("tools"):
        raise HTTPException(400, "此服務不支援 tools / function calling")
    # extra_body 的 options / think 是給 Ollama 的參數，這裡直接忽略
    if body.get("stream"):
        return StreamingResponse(generate_stream(body), media_type="text/event-stream")
    try:
        from starlette.concurrency import run_in_threadpool

        return JSONResponse(await run_in_threadpool(generate, body))
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[LoRA] 生成失敗 ({type(e).__name__}): {e}")
        raise HTTPException(500, f"生成失敗：{e}")


def main():
    global MODEL, PROCESSOR, DEVICE
    MODEL, PROCESSOR, DEVICE = load_model()
    print(f"[LoRA] 服務啟動：http://{HOST}:{PORT}/v1  (model={MODEL_NAME})")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
