#!/bin/bash
# =============================================================================
# 高贝塔股池 5min Kronos 微调 - CV 训练流水线
# =============================================================================
#
# 用法:
#   bash finetune/run_cv.sh [stage] [fold]
#
# stage:
#   preprocess  - 仅数据预处理
#   train       - 训练 (需指定 fold: 0/1/2/3/full)
#   train-all   - 训练全部4折 (顺序执行)
#   train-final - 训练最终全量模型 (fold=full)
#   check       - 验证某折数据
#   all         - 全流程: preprocess → 4折训练 → 最终模型
#
# 环境变量:
#   NGPU        - GPU 数量 (默认 4)
#   KRONOS_FOLD - 单折训练时指定 (被参数 fold 覆盖)
#
# 示例:
#   bash finetune/run_cv.sh preprocess
#   bash finetune/run_cv.sh train 0        # 训练fold0
#   bash finetune/run_cv.sh train-all      # 训练4折
#   bash finetune/run_cv.sh all            # 全流程
# =============================================================================

set -euo pipefail

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# 环境
NGPU=${NGPU:-4}
PYTHON=/home/zxh/miniconda3/envs/kronos/bin/python
TORCHRUN="$PYTHON -m torch.distributed.run"

# DDP 参数 (standalone 模式)
DDP_FLAGS="--standalone --nproc_per_node=$NGPU"

STAGE=${1:-all}
FOLD=${2:-}

echo "============================================================"
echo "高贝塔 5min Kronos CV 训练"
echo "  stage: $STAGE"
echo "  fold:  ${FOLD:-未指定}"
echo "  NGPU:  $NGPU"
echo "============================================================"

# -----------------------------------------------------------------------------
# 阶段0: 数据预处理
# -----------------------------------------------------------------------------
run_preprocess() {
    echo "[STAGE] 数据预处理 (1min→5min聚合 + 4折CV切分)..."
    $PYTHON preprocess_5min.py
    echo "[STAGE] 数据验证..."
    for f in 0 1 2 3; do
        $PYTHON check_data.py --fold $f
    done
    $PYTHON check_data.py --fold full
}

# -----------------------------------------------------------------------------
# 阶段1: 单折训练 (仅 predictor, tokenizer 用预训练的)
# -----------------------------------------------------------------------------
run_train_fold() {
    local fold=$1
    echo "[STAGE] 训练 fold=$fold (predictor only, tokenizer=pretrained)..."

    KRONOS_CONFIG=config_highbeta KRONOS_FOLD=$fold \
    HF_HUB_OFFLINE=1 \
    $TORCHRUN $DDP_FLAGS train_predictor.py

    echo "[STAGE] fold=$fold 训练完成"
}

# -----------------------------------------------------------------------------
# 主调度
# -----------------------------------------------------------------------------
case $STAGE in
    preprocess)
        run_preprocess
        ;;
    train)
        if [ -z "$FOLD" ]; then
            echo "❌ train 需要指定 fold: bash run_cv.sh train 0"
            exit 1
        fi
        run_train_fold "$FOLD"
        ;;
    train-all)
        for f in 0 1 2 3; do
            run_train_fold "$f"
        done
        echo "[STAGE] 全部4折训练完成。请检查各折 val_loss 决定最终超参。"
        ;;
    train-final)
        run_train_fold "full"
        ;;
    check)
        FOLD=${FOLD:-0}
        $PYTHON check_data.py --fold "$FOLD"
        ;;
    all)
        run_preprocess
        for f in 0 1 2 3; do
            run_train_fold "$f"
        done
        echo "[STAGE] 4折训练完成,开始最终全量模型训练..."
        run_train_fold "full"
        echo "[STAGE] ✅ 全流程完成"
        ;;
    *)
        echo "未知 stage: $STAGE"
        echo "可用: preprocess | train | train-all | train-final | check | all"
        exit 1
        ;;
esac
