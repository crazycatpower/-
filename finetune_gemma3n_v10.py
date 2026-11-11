# -*- coding: utf-8 -*-
"""
v10：用 LoRA 微調 google/gemma-3n-E2B-it，資料是 finetune_data_v10/（build_finetune_dataset_v10.py 產生）。

和 v1~v9 的差別（為什麼之前卡關，見 audit_vlm_common.py 檔頭）：
  1. 訓練時的 forward 跟推論是同一個模型。v1~v9 訓練時 cache 被關掉，第 20~29 層沒有共用 KV，
     等於在另一個模型上訓練，換回推論就幾乎沒效果。
  2. 不量化，bf16 直接訓練（per-layer embedding 留在 CPU、拿掉音訊編碼器，GPU 約 5GB 起跳）。
     部署時 llm_server.py 也是 bf16，少一層 4-bit 與 bf16 的落差。
  3. 不用 TRL，自己寫訓練迴圈：答案以外的位置不過 lm_head（省 VRAM）、訓練中定期在 dev 上量
     「true/false 那一格」的準確率，留下 dev 最好的 checkpoint，不再拿 val57 挑 checkpoint。
  4. LoRA 只掛在語言模型上（舊版連用不到的音訊編碼器也掛了），而且第 20~29 層的 k_proj/v_proj 不掛，
     因為那幾層的 KV 是共用來的，這兩個投影根本不會被用到。

用法：
  python finetune_gemma3n_v10.py            # → gemma3n-e2b-audit-lora-v10/{best,last}
  python finetune_gemma3n_v10.py --item     # prompt 多給稽核項目 → gemma3n-e2b-audit-lora-v10-item/
中斷後重跑同一個指令會從 last/ 接著訓練。
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

DATA_DIR = C.BASE_DIR / "finetune_data_v10"
SEED = 42
EPOCHS = 4
ACCUM = 8                 # 每 8 張圖更新一次權重
LR = 1e-4
LR_FLOOR = 0.1            # cosine 衰減到 LR 的幾成
WARMUP_STEPS = 10
MAX_GRAD_NORM = 1.0
EVAL_EVERY = 20           # 每幾次權重更新在 dev 上量一次
VIOLATION_OVERSAMPLE = 2  # 缺失樣本每個 epoch 出現幾次（v6 起的設定：漏報比誤報嚴重）
# YOLO 清單特別長的圖（約 3% 的樣本）訓練時略過，評測不受影響。3400 token 單獨跑放得進 12GB，但桌面程式
# 同時佔用顯存時會溢出到共享記憶體——不會報錯，只是一步從 40 秒變成一小時以上（第一次訓練就卡在這裡）。
MAX_TRAIN_TOKENS = 2000
LORA = dict(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=(r"model\.language_model\.layers\.\d+\.(self_attn\.(q|o)_proj|mlp\.(gate|up|down)_proj)"
                    r"|model\.language_model\.layers\.1?\d\.self_attn\.(k|v)_proj"),
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def epoch_rows(groups, epoch):
    """這個 epoch 要看的樣本：每張圖輪流用不同版本（原圖、aug0、aug1）；缺失樣本出現
    VIOLATION_OVERSAMPLE 次，而且每次是不同版本。"""
    rows = []
    for versions in groups:
        repeats = 1 if json.loads(versions[0]["response"])["compliant"] else VIOLATION_OVERSAMPLE
        rows += [versions[(epoch + k) % len(versions)] for k in range(repeats)]
    random.Random(SEED + epoch).shuffle(rows)
    return rows


def lr_at(step, total_steps):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LR * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", action="store_true", help="prompt 最後多給「本張照片的稽核項目」")
    ap.add_argument("--out", help="輸出資料夾名稱")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-steps", type=int, help="只跑這麼多次權重更新（試跑用）")
    args = ap.parse_args()
    out_dir = C.BASE_DIR / (args.out or "gemma3n-e2b-audit-lora-v10" + ("-item" if args.item else ""))
    best_dir, last_dir = out_dir / "best", out_dir / "last"

    def prompt_of(row):
        return row["prompt"] + ("\n\n" + C.ITEM_HINT.format(row["item"]) if args.item else "")

    by_group = {}
    for row in read_jsonl(DATA_DIR / "train.jsonl"):
        by_group.setdefault(row["group"], []).append(row)
    groups = [sorted(v, key=lambda r: r["version"]) for _, v in sorted(by_group.items())]
    dev = read_jsonl(DATA_DIR / "dev.jsonl")
    steps_per_epoch = math.ceil(len(epoch_rows(groups, 0)) / ACCUM)
    total_steps = min(args.max_steps or 10**9, steps_per_epoch * args.epochs)

    torch.manual_seed(SEED)
    model, processor = C.load_model()
    gemma = model
    model = get_peft_model(gemma, LoraConfig(**LORA))
    # gradient checkpointing：不存每一層的 activation，backward 時再重算。prompt 裡的 YOLO 清單長的有三千多 token，
    # 不開的話 12GB 放不下。要在 get_peft_model 之後才開，否則 PEFT 會呼叫 enable_input_require_grads()，
    # 而 transformers 5.0.0.dev0 的那個方法在音訊／視覺編碼器上會丟 NotImplementedError。
    gemma.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
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

    @torch.no_grad()
    def evaluate_dev():
        """dev 上的 teacher-forced 指標。decision_* 只看答案裡 true/false 那一格——生成時整張圖的判定就是由它決定。"""
        model.eval()
        loss_sum = tokens = correct = 0
        nll = 0.0
        confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for row in dev:
            batch, labels = C.encode(processor, C.load_image(C.BASE_DIR / row["image"]), prompt_of(row), row["response"])
            logits, targets = C.answer_logits(model, batch, labels)
            loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            tokens += len(targets)
            p_violation = torch.softmax(logits[dec_pos, [true_id, false_id]], 0)[1].item()
            gt = targets[dec_pos].item() == false_id
            nll -= math.log(max(p_violation if gt else 1 - p_violation, 1e-6))
            correct += (p_violation >= 0.5) == gt
            confusion[("t" if (p_violation >= 0.5) == gt else "f") + ("p" if p_violation >= 0.5 else "n")] += 1
        return {"dev_loss": round(loss_sum / tokens, 4), "decision_acc": round(correct / len(dev), 4),
                "decision_nll": round(nll / len(dev), 4), **confusion}

    def save(directory, extra=None):
        model.save_pretrained(str(directory))
        processor.save_pretrained(str(directory))  # llm_server.py 從 adapter 資料夾讀 processor
        if extra is not None:
            torch.save(extra, directory / "state.pt")

    def log(record):
        state["history"].append(record)
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if state["step"] == 0:
        log({"step": 0, "epoch": 0, **evaluate_dev()})  # 還沒訓練的底座，當作對照

    t0 = time.time()
    window_loss = window_tokens = skipped = 0
    too_long = set()
    for epoch in range(state["step"] // steps_per_epoch, args.epochs):
        rows = epoch_rows(groups, epoch)
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
                    batch, labels = C.encode(processor, C.load_image(C.BASE_DIR / row["image"]), prompt_of(row), row["response"])
                    if batch["input_ids"].shape[1] > MAX_TRAIN_TOKENS:
                        too_long.add(row["image"])
                        continue
                    logits, targets = C.answer_logits(model, batch, labels)
                    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
                    loss.backward()
                    tokens += len(targets)
                    step_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                # 偵測清單特別長的圖偶爾放不下。backward 做到一半的梯度不完整，這一步整個不更新，但訓練繼續
                skipped += 1
                tokens = step_loss = 0
                batch = labels = logits = targets = loss = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            if tokens:
                # 每個答案 token 權重相同（合規的答案只有十來個 token，缺失的有幾十個）
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(tokens)
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()  # 每張圖長度不同，不歸還的話保留區會越墊越高
            window_loss += step_loss
            window_tokens += tokens
            state["step"] += 1

            done = state["step"] == total_steps
            if state["step"] % 10 == 0 or done:
                elapsed = time.time() - t0
                print(f"[step {state['step']}/{total_steps}] epoch {state['step'] / steps_per_epoch:.2f} "
                      f"train_loss {window_loss / max(window_tokens, 1):.4f} lr {lr_at(state['step'] - 1, total_steps):.2e} "
                      f"{elapsed / 60:.0f} 分鐘" + (f" 因顯存不足跳過 {skipped} 步" if skipped else ""), flush=True)
            if state["step"] % EVAL_EVERY == 0 or done:
                metrics = evaluate_dev()
                record = {"step": state["step"], "epoch": round(state["step"] / steps_per_epoch, 2),
                          "train_loss": round(window_loss / max(window_tokens, 1), 4), **metrics}
                window_loss = window_tokens = 0
                # 先比判定準確率；48 張的準確率一格是 2%，同分時取 true/false 那一格的 NLL 較低者
                score = (metrics["decision_acc"], -metrics["decision_nll"])
                if state["best"] is None or score > tuple(state["best"]["score"]):
                    state["best"] = {"score": score, **record}
                    save(best_dir)
                    record["best"] = True
                log(record)
                save(last_dir, {**state, "optimizer": optimizer.state_dict()})

    (out_dir / "summary.json").write_text(json.dumps(
        {"args": vars(args), "lora": LORA, "epochs": args.epochs, "lr": LR, "accum": ACCUM,
         "train_images": len(groups), "dev_images": len(dev), "skipped_too_long": sorted(too_long),
         "oom_steps": skipped, "best": state["best"], "history": state["history"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", best_dir, json.dumps(state["best"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
