#!/bin/bash
# 5min 特征提取后台运行脚本
#
# 单卡共存模式:sglang 占满 GPU,但 Kronos 推理只需 ~3GB
# NCCL 隔离标志避免与 sglang 冲突
#
# 用法:
#   bash run_extract.sh finetune   # 跑微调版
#   bash run_extract.sh base       # 跑预训练版
#   bash run_extract.sh smoke      # smoke test (10股×5天)

set -e

CHECKPOINT=${1:-finetune}
cd /home/zxh/projects/Kronos/finetune/feature_extraction

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

PY=/home/zxh/miniconda3/envs/kronos/bin/python

LOG_DIR=/home/zxh/projects/Kronos/finetune/feature_extraction/output/logs
mkdir -p "$LOG_DIR"

if [ "$CHECKPOINT" == "smoke" ]; then
    echo "Running smoke test..."
    $PY extract_features.py --checkpoint finetune --max-stocks 10 --max-days 5
    exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/extract_${CHECKPOINT}_${TS}.log"

echo "Starting full extraction: checkpoint=$CHECKPOINT"
echo "Log: $LOG_FILE"
echo "Estimated time: ~15 hours (61500 predictions × ~850ms)"

nohup $PY extract_features.py --checkpoint "$CHECKPOINT" > "$LOG_FILE" 2>&1 &
PID=$!
echo "PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
echo "Kill:    kill $PID"

# 记录 PID 便于后续管理
echo "$PID" > "$LOG_DIR/extract_${CHECKPOINT}.pid"
