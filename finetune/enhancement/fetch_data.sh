#!/bin/bash
# =============================================================================
# 高贝塔指增方案 - 数据拉取总入口
# =============================================================================
#
# 拉取:
#   1. 883926.TI 指数日线行情(fetch_index_daily.py)
#   2. 883926.TI 历史成分股快照(fetch_universe.py)
#
# 依赖:
#   - conda 环境 ifind(/home/zxh/miniconda3/envs/ifind)
#   - IfindClient 库(/home/zxh/projects/2.qlib_ifind_hot_concept)
#   - iFinD 账号(secrets/ifind_token.json,active 账号需 refresh_token 有效)
#
# 用法:
#   bash finetune/enhancement/fetch_data.sh                       # 拉全部
#   bash finetune/enhancement/fetch_data.sh index                 # 只拉指数
#   bash finetune/enhancement/fetch_data.sh universe              # 只拉成分股
#   bash finetune/enhancement/fetch_data.sh universe --resume     # 断点续拉
#   bash finetune/enhancement/fetch_data.sh index --start 2026-07-10 --end 2026-07-17
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=/home/zxh/miniconda3/envs/ifind/bin/python
IFIND_LIB=/home/zxh/projects/2.qlib_ifind_hot_concept
export PYTHONPATH="$IFIND_LIB:${PYTHONPATH:-}"

DATA_DIR="$PROJECT_ROOT/finetune/data/enhancement"
mkdir -p "$DATA_DIR"

STAGE="${1:-all}"
if [[ "$STAGE" == "index" || "$STAGE" == "universe" ]]; then
    shift  # 去掉 stage,剩下参数透传给 python 脚本
fi

echo "============================================================"
echo "高贝塔指增 - 数据拉取 (stage=$STAGE)"
echo "============================================================"

case "$STAGE" in
    index)
        $PYTHON finetune/enhancement/fetch_index_daily.py "$@"
        ;;
    universe)
        $PYTHON finetune/enhancement/fetch_universe.py "$@"
        ;;
    all)
        echo "[1/2] 指数日线..."
        $PYTHON finetune/enhancement/fetch_index_daily.py
        echo
        echo "[2/2] 成分股历史快照..."
        $PYTHON finetune/enhancement/fetch_universe.py
        echo
        echo "✅ 全部完成。产物:"
        ls -la "$DATA_DIR"/*.csv 2>/dev/null
        ;;
    *)
        echo "未知 stage: $STAGE"
        echo "可用: index | universe | all"
        exit 1
        ;;
esac
