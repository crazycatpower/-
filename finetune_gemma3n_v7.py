# -*- coding: utf-8 -*-
"""
用 QLoRA 微調 google/gemma-3n-E2B-it，訓練資料來自 finetune_data/{train,val}.jsonl
（由 build_finetune_dataset.py 產生）。

用法:
  python finetune_gemma3n.py
"""
import io
import json
import os
from pathlib import Path

# v5：step 15 在 12GB 卡上 CUDA OOM（還會溢出到 shared memory 讓 step 從 90s 爬到 155s）。
# expandable_segments 降低顯存碎片；真正的解法是下面把 gradient_checkpointing 打開。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3nForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "google/gemma-3n-E2B-it"
OUT_DIR = BASE_DIR / "gemma4-e2b-audit-lora-v7"
DATA_DIR = BASE_DIR / "finetune_data"
WARM_START_PATH = None  # 全新重跑，不接續容易卡重複迴圈的舊權重

# v7 對照組：單變量 ablation，補上 v6 缺的那一格。資料沿用 v6 的過採樣 ×2
# （finetune_data/，620 筆），但 epoch 維持 v5 的 2 輪，不加到 3。
# 對照 v5-e3（無過採樣、epoch 3，結果比 v5 差）跟 v6（過採樣+epoch3，recall 0.634），
# 用來判斷 epoch 2→3 這一項是否真的有獨立貢獻，還是只是配合資料變多在收斂。

# v5：與 test_100_real_flow.py / test_100_noyolo_strict.py 的 SYSTEM_PROMPT 逐字一致，
# 讓訓練與評測/推論的 system prompt、輸出 schema 完全對齊（見 build_finetune_dataset.py v5 說明）。
SYSTEM_PROMPT = (
    "你是專業的工地職安稽核員。分析照片後只輸出 JSON："
    '{"compliant": true/false, "findings": [{"item":"項目名稱","result":"缺失描述"}]}'
    "，符合規定時 findings 為空陣列。"
)


def build_conversation(example):
    # 只存路徑，不在 datasets(Arrow) 裡放 PIL 物件（Arrow 會把它序列化成 dict 而壞掉），
    # 實際開圖延後到 collate_fn。
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image_path": example["image"]},
            {"type": "text", "text": example["prompt"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": example["response"]}]},
    ]
    return {"messages": messages}


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["prediction_coefs", "correction_coefs"],
    )

    hf_token = os.getenv("HF_TOKEN") or None
    # 遠端 chat_template.json 這個(可能根本不存在的)檔案在新版 transformers 會被
    # 誤判成 gated-repo 401，本地已有完整快取，改成強制只吃本地檔案繞開網路請求。
    local_snapshot = next(
        (Path.home() / ".cache" / "huggingface" / "hub"
         / "models--google--gemma-3n-E2B-it" / "snapshots").glob("*"),
        None,
    )
    load_id = str(local_snapshot) if local_snapshot else MODEL_ID
    load_kwargs = {"local_files_only": True} if local_snapshot else {"token": hf_token}
    print(f"[model-load] {load_id} ({'local' if local_snapshot else 'remote'})")

    processor = AutoProcessor.from_pretrained(load_id, **load_kwargs)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        load_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        **load_kwargs,
    )
    model.config.use_cache = False

    prior_checkpoints = sorted(OUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if OUT_DIR.exists() else []
    usable_checkpoint = None
    for ckpt in reversed(prior_checkpoints):
        if (ckpt / "trainer_state.json").exists():
            usable_checkpoint = ckpt
            break

    if usable_checkpoint:
        print(f"[resume] 從 {usable_checkpoint}（含 trainer_state）完整續跑")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(usable_checkpoint), is_trainable=True)
    elif prior_checkpoints:
        # 最新 checkpoint 缺 trainer_state.json（例如訓練途中斷電），
        # 無法用 resume_from_checkpoint；改成只載入 LoRA 權重當起點，重新起一輪 optimizer/LR。
        latest = prior_checkpoints[-1]
        print(f"[warm-start] {latest} 缺 trainer_state.json，改成只載入權重、重新起 optimizer/LR")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(latest), is_trainable=True)
        usable_checkpoint = None
    elif WARM_START_PATH and WARM_START_PATH.exists():
        print(f"[warm-start] 從斷電備份 {WARM_START_PATH} 只載入權重，重新起 optimizer/LR")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(WARM_START_PATH), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # gradient checkpointing 對 4-bit 凍結底座＋LoRA 是必需的：沒有它 12GB 放不下
    # 896px 影像的 activation。quantized 底座的輸入不帶 grad，要手動打開讓
    # checkpoint 的重算能把梯度回傳到 LoRA。
    #
    # transformers 5.0.0.dev0 的 enable_input_require_grads() 會遍歷所有子模組、
    # 對每個 PreTrainedModel 呼叫 get_input_embeddings()，遇到 Gemma3nAudioEncoder /
    # 視覺塔就丟 NotImplementedError。Trainer 在 gradient_checkpointing_enable() 裡
    # 對 PEFT 模型也會自己呼叫一次——所以直接把這個方法換成「只掛文字 embedding」的版本，
    # 我方與 Trainer 的呼叫都會走這條安全路徑。
    def _text_only_require_grads(*_args, **_kwargs):
        emb = model.get_input_embeddings()  # -> Gemma3nModel.language_model.embed_tokens
        model._require_grads_hook = emb.register_forward_hook(
            lambda _m, _in, out: out.requires_grad_(True)
        )

    # 三個層級都換掉：PeftModel、LoraModel(base_model)、底層 transformers 模型
    # （Trainer 的 gradient_checkpointing_enable 是在底層模型上呼叫 enable_input_require_grads）。
    for obj in {model, getattr(model, "base_model", None), getattr(model, "get_base_model", lambda: None)()}:
        if obj is not None:
            obj.enable_input_require_grads = _text_only_require_grads
    _text_only_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={
        "train": str(DATA_DIR / "train.jsonl"),
        "validation": str(DATA_DIR / "val.jsonl"),
    })
    ds = ds.map(build_conversation, remove_columns=["image", "prompt", "response"])

    _MAX_SIDE = 896  # 原圖有些高達 4624x3468（16MP），沒縮圖直接餵進 vision tower 會炸顯存

    def _resize(img):
        w, h = img.size
        if max(w, h) <= _MAX_SIDE:
            return img
        scale = _MAX_SIDE / max(w, h)
        return img.resize((int(w * scale), int(h * scale)))

    def _load_img(c):
        # datasets 有時會把我們的 image_path 字串自動轉成 {"path":..,"bytes":..} 結構
        val = c.get("image_path") or c.get("image")
        if isinstance(val, dict):
            if val.get("bytes"):
                img = Image.open(io.BytesIO(val["bytes"])).convert("RGB")
            else:
                img = Image.open(val["path"]).convert("RGB")
        else:
            img = Image.open(val).convert("RGB")
        return _resize(img)

    # v5：completion-only 損失。舊版只遮 pad，等於逼模型連 system prompt、
    # 那一大段半模板化的 YOLO 偵測清單、甚至 image soft token 都要「生成」一遍——
    # 既浪費容量，模板化文字還會拉高生成時卡重複迴圈的機率（正是 v4 在對付的病灶）。
    # 這裡把 assistant 回答之前的所有 token 全部設成 -100，只在真正的 JSON 答案上算 loss。
    _IMAGE_TOKEN_ID = getattr(processor, "image_token_id", 262145)
    _RESP_TEMPLATE = processor.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)

    def _find_last_subseq(ids, pat):
        n, m = len(ids), len(pat)
        for i in range(n - m, -1, -1):
            if ids[i:i + m] == pat:
                return i
        return -1

    def collate_fn(examples):
        texts = [processor.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
                  for e in examples]
        images = [
            [_load_img(c) for m in e["messages"] for c in m["content"] if c.get("type") == "image"]
            for e in examples
        ]
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == _IMAGE_TOKEN_ID] = -100
        for row in labels:
            cut = _find_last_subseq(row.tolist(), _RESP_TEMPLATE)
            if cut >= 0:
                row[: cut + len(_RESP_TEMPLATE)] = -100
        batch["labels"] = labels
        return batch

    sft_config = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=2,  # v7：過採樣但只訓2輪，拆解epoch3是否必要
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=27,  # 620筆*2epoch/8 約155步，維持跟v5類似的存檔密度
        save_total_limit=2,
        save_only_model=True,
        logging_steps=10,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate_fn,
    )

    trainer.train(resume_from_checkpoint=str(usable_checkpoint) if usable_checkpoint else None)
    trainer.save_model(str(OUT_DIR / "final"))
    processor.save_pretrained(str(OUT_DIR / "final"))
    print("DONE", OUT_DIR / "final")


if __name__ == "__main__":
    main()
