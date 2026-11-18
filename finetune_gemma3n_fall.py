# -*- coding: utf-8 -*-
"""
墜落防止專用微調（v12-fall）：從 v10 best 接著訓練，資料是 finetune_data_fall/（build_finetune_dataset_fall.py）。

和 v10 的差別：
  1. 起點是 v10 best 的 LoRA 權重，不是底座。學習率減半（5e-5），只跑 3 個 epoch。
  2. 每個 epoch：墜落照片全部（版本輪替）＋ 同樣張數、每個 epoch 重抽的其他類別照片（v10 train），
     避免其他類別退步。缺失不加倍（v11 實驗顯示加倍與否差別不大）。
  3. 每筆樣本 50% 機率在 prompt 後面附「本張照片的稽核項目」，所以給不給稽核項目都能用。
  4. dev 分兩部分回報：墜落 dev（12 張）與其他類別 dev（v10 dev 去掉墜落照片，42 張），
     依合計判定準確率挑 checkpoint；dev 評估時不給稽核項目。

用法：python finetune_gemma3n_fall.py   → gemma3n-e2b-audit-lora-fall/{best,last}
中斷後重跑同一個指令會從 last/ 接著訓練。模型載入、LoRA 設定、loss 計算與 v10 相同。
"""
import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

import audit_vlm_common as C
from finetune_gemma3n_v10 import LORA, MAX_GRAD_NORM, MAX_TRAIN_TOKENS, read_jsonl

FALL_DIR = C.BASE_DIR / "finetune_data_fall"
V10_DIR = C.BASE_DIR / "finetune_data_v10"
INIT_ADAPTER = C.BASE_DIR / "gemma3n-e2b-audit-lora-v10" / "best"
SEED = 42
EPOCHS = 3
ACCUM = 8
LR = 5e-5
LR_FLOOR = 0.1
WARMUP_STEPS = 5
EVAL_EVERY = 6
REPLAY_RATIO = 1.0      # 每個 epoch 抽幾倍於墜落照片張數的其他類別照片
ITEM_HINT_PROB = 0.5


def lr_at(step, total_steps):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LR * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress)))


def grouped(rows):
    by = {}
    for row in rows:
        by.setdefault(row["group"], []).append(row)
    return [sorted(v, key=lambda r: r["version"]) for _, v in sorted(by.items())]


def epoch_rows(fall_groups, replay_groups, epoch):
    rng = random.Random(SEED + epoch)
    n_replay = round(len(fall_groups) * REPLAY_RATIO)
    picked = fall_groups + rng.sample(replay_groups, n_replay)
    rows = []
    for versions in picked:
        row = versions[epoch % len(versions)]
        rows.append({**row, "with_item": rng.random() < ITEM_HINT_PROB})
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gemma3n-e2b-audit-lora-fall")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-steps", type=int, help="只跑這麼多次權重更新（試跑用）")
    args = ap.parse_args()
    out_dir = C.BASE_DIR / args.out
    best_dir, last_dir = out_dir / "best", out_dir / "last"

    def prompt_of(row, with_item=False):
        return row["prompt"] + ("\n\n" + C.ITEM_HINT.format(row["item"]) if with_item else "")

    fall_train = read_jsonl(FALL_DIR / "train.jsonl")
    fall_dev = read_jsonl(FALL_DIR / "dev.jsonl")
    fall_groups = grouped(fall_train)
    fall_names = {g[0]["group"] for g in fall_groups} | {r["group"] for r in fall_dev}
    replay_groups = [g for g in grouped(read_jsonl(V10_DIR / "train.jsonl")) if g[0]["group"] not in fall_names]
    general_dev = [r for r in read_jsonl(V10_DIR / "dev.jsonl") if r["group"] not in fall_names]
    dev = [("fall", r) for r in fall_dev] + [("general", r) for r in general_dev]
    steps_per_epoch = math.ceil(len(epoch_rows(fall_groups, replay_groups, 0)) / ACCUM)
    total_steps = min(args.max_steps or 10**9, steps_per_epoch * args.epochs)
    print(f"墜落 train {len(fall_groups)} 張、replay 池 {len(replay_groups)} 張、dev 墜落 {len(fall_dev)} / 其他 {len(general_dev)}，"
          f"每 epoch {steps_per_epoch} 步，共 {total_steps} 步", flush=True)

    torch.manual_seed(SEED)
    model, processor = C.load_model()
    gemma = model
    model = get_peft_model(gemma, LoraConfig(**LORA))
    gemma.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    dec_pos, true_id, false_id = C.decision_token_ids(processor.tokenizer)

    state = {"step": 0, "best": None, "history": []}
    if (last_dir / "state.pt").exists():
        saved = torch.load(last_dir / "state.pt", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(last_dir / "adapter_model.safetensors")))
        optimizer.load_state_dict(saved.pop("optimizer"))
        state = saved
        print(f"[resume] 從 {last_dir} 的第 {state['step']} 步接著訓練", flush=True)
    else:
        result = set_peft_model_state_dict(model, load_file(str(INIT_ADAPTER / "adapter_model.safetensors")))
        missing = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
        assert not missing, f"v10 adapter 缺少 LoRA 權重：{missing[:3]}"
        print(f"[init] 載入 {INIT_ADAPTER}", flush=True)

    @torch.no_grad()
    def evaluate_dev():
        model.eval()
        stats = {k: {"n": 0, "correct": 0, "nll": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0} for k in ("fall", "general")}
        for part, row in dev:
            batch, labels = C.encode(processor, C.load_image(C.BASE_DIR / row["image"]), prompt_of(row), row["response"])
            logits, targets = C.answer_logits(model, batch, labels)
            p_violation = torch.softmax(logits[dec_pos, [true_id, false_id]], 0)[1].item()
            gt = targets[dec_pos].item() == false_id
            s = stats[part]
            s["n"] += 1
            s["nll"] -= math.log(max(p_violation if gt else 1 - p_violation, 1e-6))
            s["correct"] += (p_violation >= 0.5) == gt
            s[("t" if (p_violation >= 0.5) == gt else "f") + ("p" if p_violation >= 0.5 else "n")] += 1
        out = {}
        for part, s in stats.items():
            out[f"{part}_acc"] = round(s["correct"] / s["n"], 4)
            out[f"{part}_nll"] = round(s["nll"] / s["n"], 4)
            out[f"{part}_conf"] = [s["tp"], s["tn"], s["fp"], s["fn"]]
        n = sum(s["n"] for s in stats.values())
        out["decision_acc"] = round(sum(s["correct"] for s in stats.values()) / n, 4)
        out["decision_nll"] = round(sum(s["nll"] for s in stats.values()) / n, 4)
        return out

    def save(directory, extra=None):
        model.save_pretrained(str(directory))
        processor.save_pretrained(str(directory))
        if extra is not None:
            torch.save(extra, directory / "state.pt")

    def log(record):
        state["history"].append(record)
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def consider(record):
        score = (record["decision_acc"], -record["decision_nll"])
        if state["best"] is None or score > tuple(state["best"]["score"]):
            state["best"] = {"score": score, **record}
            save(best_dir)
            record["best"] = True

    if state["step"] == 0:
        record = {"step": 0, "epoch": 0, **evaluate_dev()}  # 起點 = v10 best
        consider(record)
        log(record)

    t0 = time.time()
    window_loss = window_tokens = skipped = 0
    too_long = set()
    for epoch in range(state["step"] // steps_per_epoch, args.epochs):
        rows = epoch_rows(fall_groups, replay_groups, epoch)
        for step_in_epoch in range(state["step"] - epoch * steps_per_epoch, steps_per_epoch):
            if state["step"] >= total_steps:
                break
            model.train()
            gemma.model.vision_tower.eval()
            for group in optimizer.param_groups:
                group["lr"] = lr_at(state["step"], total_steps)
            tokens = step_loss = 0
            try:
                for row in rows[step_in_epoch * ACCUM:(step_in_epoch + 1) * ACCUM]:
                    batch, labels = C.encode(processor, C.load_image(C.BASE_DIR / row["image"]),
                                             prompt_of(row, row["with_item"]), row["response"])
                    if batch["input_ids"].shape[1] > MAX_TRAIN_TOKENS:
                        too_long.add(row["image"])
                        continue
                    logits, targets = C.answer_logits(model, batch, labels)
                    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
                    loss.backward()
                    tokens += len(targets)
                    step_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                tokens = step_loss = 0
                batch = labels = logits = targets = loss = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            if tokens:
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(tokens)
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            window_loss += step_loss
            window_tokens += tokens
            state["step"] += 1

            done = state["step"] == total_steps
            print(f"[step {state['step']}/{total_steps}] loss {step_loss / max(tokens, 1):.4f} "
                  f"{(time.time() - t0) / 60:.0f} 分鐘", flush=True)
            if state["step"] % EVAL_EVERY == 0 or done:
                record = {"step": state["step"], "epoch": round(state["step"] / steps_per_epoch, 2),
                          "train_loss": round(window_loss / max(window_tokens, 1), 4), **evaluate_dev()}
                window_loss = window_tokens = 0
                consider(record)
                log(record)
                save(last_dir, {**state, "optimizer": optimizer.state_dict()})

    (out_dir / "summary.json").write_text(json.dumps(
        {"args": vars(args), "init": str(INIT_ADAPTER.relative_to(C.BASE_DIR)), "lora": LORA, "lr": LR,
         "accum": ACCUM, "replay_ratio": REPLAY_RATIO, "item_hint_prob": ITEM_HINT_PROB,
         "fall_train_images": len(fall_groups), "dev": {"fall": len(fall_dev), "general": len(general_dev)},
         "skipped_too_long": sorted(too_long), "oom_steps": skipped, "best": state["best"],
         "history": state["history"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", best_dir, json.dumps(state["best"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
