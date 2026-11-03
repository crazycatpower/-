#!/bin/bash
# 自動化流程：等 noyolo/e3 兩組訓練都跑完 -> 依序評測 -> 彙整結果。
# 由現有的 noyolo->e3 排程 wrapper 接手之後啟動，本腳本只負責「等 e3 訓練結束 -> 跑評測 -> 彙整」。
set -u
cd "C:/Users/q1q2q/Downloads/0603-20260629T233729Z-3-001/0603"
LOG=pipeline.log
echo "[pipeline] $(date) 開始等待 e3 訓練結束（輪詢 train_v5e3.log，最多等 8 小時）" >> "$LOG"

# 直接盯 train_v5e3.log 的完成/失敗訊號，不靠「process 是否存在」判斷——
# 避免 noyolo 結束到 e3 啟動之間那個短暫空窗被誤判成「訓練已結束」。
i=0
status=""
while [ $i -lt 1440 ]; do
  if [ -f train_v5e3.log ] && grep -q "^DONE " train_v5e3.log; then status="done"; break; fi
  if [ -f train_v5e3.log ] && grep -qE "Traceback|OutOfMemoryError" train_v5e3.log; then status="failed"; break; fi
  sleep 20
  i=$((i+1))
done

if [ "$status" != "done" ]; then
  echo "[pipeline] e3 訓練未正常完成（status=$status，等了 $i 次），停止流程" >> "$LOG"
  exit 1
fi
echo "[pipeline] $(date) e3 訓練完成" >> "$LOG"

if [ ! -d "gemma4-e2b-audit-lora-v5-e3/final" ] || [ ! -d "gemma4-e2b-audit-lora-v5-noyolo/final" ]; then
  echo "[pipeline] adapter 資料夾缺失，停止流程" >> "$LOG"
  exit 1
fi

echo "[pipeline] $(date) 兩組 adapter 都存在，開始評測 v5-noyolo" >> "$LOG"
python -u test_100_real_flow_v5noyolo.py >> eval_v5noyolo.log 2>&1
echo "[pipeline] $(date) v5-noyolo 評測完成，開始評測 v5-e3" >> "$LOG"
python -u test_100_real_flow_v5e3.py >> eval_v5e3.log 2>&1
echo "[pipeline] $(date) v5-e3 評測完成，彙整結果" >> "$LOG"

python -u - <<'PYEOF' >> "$LOG" 2>&1
import json
files = {
    "v1": "test_100_result.json",
    "v5": "test_100_v5_result.json",
    "v6": "test_100_v6_result.json",
    "v5-noyolo": "test_100_v5noyolo_result.json",
    "v5-e3": "test_100_v5e3_result.json",
}
out = {}
for name, f in files.items():
    try:
        out[name] = json.load(open(f, encoding="utf-8"))["summary"]
    except Exception as e:
        out[name] = {"error": str(e)}
json.dump(out, open("ablation_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("=== ablation_summary.json ===")
for name, s in out.items():
    print(name, s)
PYEOF

echo "[pipeline] $(date) 全部完成，見 ablation_summary.json" >> "$LOG"
touch ALL_DONE.marker
