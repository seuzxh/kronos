#!/bin/bash
# =============================================================================
# 高贝塔指增 DAILY Loop - 训练流水线编排
# =============================================================================
#
# 用法:
#   bash finetune/enhancement/run_daily.sh <stage> [fold]
#
# stage:
#   preprocess  - 日频预处理(个股日线 + 指数对齐 → 4折CV pickle)
#   smoke       - 冒烟测试(单 GPU,验证训练管线 ~1分钟)
#   train       - 训练单折 (需指定 fold: 0/1/2/3/full)
#   train-all   - 训练全部 4 折
#   train-final - 训练最终全量模型
#   backtest    - 单折回测(需指定 fold)
#   backtest-all- 全部折回测 + 门禁汇总
#   all         - 全流程: preprocess → smoke → 4折训练 → full → 回测汇总
#
# 环境变量:
#   NGPU        - GPU 数量(默认 4)
#   KRONOS_FOLD - 单折训练时指定(被参数 fold 覆盖)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$FINETUNE_DIR/.." && pwd)"
cd "$FINETUNE_DIR"

# 环境
NGPU=${NGPU:-4}
PYTHON=/home/zxh/miniconda3/envs/kronos/bin/python
TORCHRUN="$PYTHON -m torch.distributed.run"
DDP_FLAGS="--standalone --nproc_per_node=$NGPU"

# NCCL 配置(sglang 共存必需,与 run_cv.sh 一致)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

STAGE=${1:-all}
FOLD=${2:-}

echo "============================================================"
echo "高贝塔指增 DAILY Loop"
echo "  stage: $STAGE  fold: ${FOLD:-未指定}  NGPU: $NGPU"
echo "============================================================"

run_preprocess() {
    echo "[STAGE] 日频预处理(个股日线 + 指数对齐)..."
    $PYTHON enhancement/preprocess_daily.py
}

run_smoke() {
    echo "[STAGE] 冒烟测试..."
    KRONOS_CONFIG=config_daily KRONOS_FOLD=0 $PYTHON enhancement/smoke_daily.py
}

run_train_fold() {
    local fold=$1
    echo "[STAGE] 训练 fold=$fold (predictor only, tokenizer=pretrained)..."
    KRONOS_CONFIG=config_daily KRONOS_FOLD=$fold \
    $TORCHRUN $DDP_FLAGS train_predictor.py
    echo "[STAGE] fold=$fold 训练完成"
}

run_backtest_fold() {
    local fold=$1
    echo "[STAGE] 回测 fold=$fold ..."
    KRONOS_CONFIG=config_daily $PYTHON enhancement/backtest_excess.py --fold $fold
}

case "$STAGE" in
    preprocess)
        run_preprocess
        ;;
    smoke)
        run_smoke
        ;;
    train)
        if [ -z "$FOLD" ]; then
            echo "❌ train 需要指定 fold: bash run_daily.sh train 0"
            exit 1
        fi
        run_train_fold "$FOLD"
        ;;
    train-all)
        for f in 0 1 2 3; do
            run_train_fold "$f"
        done
        echo "[STAGE] 全部 4 折训练完成"
        ;;
    train-final)
        run_train_fold "full"
        ;;
    backtest)
        if [ -z "$FOLD" ]; then
            echo "❌ backtest 需要指定 fold: bash run_daily.sh backtest 0"
            exit 1
        fi
        run_backtest_fold "$FOLD"
        ;;
    backtest-all)
        KRONOS_CONFIG=config_daily $PYTHON enhancement/backtest_excess.py --all
        ;;
    all)
        run_preprocess
        run_smoke
        for f in 0 1 2 3; do
            run_train_fold "$f"
        done
        echo "[STAGE] 4 折训练完成,开始最终全量模型..."
        run_train_fold "full"
        echo "[STAGE] 回测 + 门禁检查..."
        KRONOS_CONFIG=config_daily $PYTHON enhancement/backtest_excess.py --all
        echo "[STAGE] ✅ 全流程完成"
        ;;
    *)
        echo "未知 stage: $STAGE"
        echo "可用: preprocess | smoke | train | train-all | train-final | backtest | backtest-all | all"
        exit 1
        ;;
esac
