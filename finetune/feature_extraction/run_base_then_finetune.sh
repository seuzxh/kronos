#!/bin/bash
# 串联提取:base 完成后自动跑 finetune,然后跑 LGBM-B 和 LGBM-C
#
# 用法(等当前 base 跑完后执行,或 nohup 后台):
#   nohup bash run_base_then_finetune.sh > /tmp/pipeline.log 2>&1 &

set -e
cd /home/zxh/projects/Kronos/finetune/feature_extraction

PY=/home/zxh/miniconda3/envs/kronos/bin/python
LGBM_PY=/home/zxh/miniconda3/envs/qlib_ifind_beta/bin/python
LGBM_DIR=/home/zxh/projects/3.qlib_ifind_beta

echo "=========================================="
echo "阶段 1:base 版特征提取(6 GPU 并行)"
echo "=========================================="
date
$PY extract_parallel.py --checkpoint base --gpus 1,2,3,4,5,6

echo ""
echo "=========================================="
echo "阶段 2:LGBM-B(base 特征)"
echo "=========================================="
date
cd $LGBM_DIR
$LGBM_PY qrun/run.py qrun/workflow_kronos_ablation_B_base.yaml 2>&1 | tee /tmp/ablation_B.log

echo ""
echo "=========================================="
echo "阶段 3:备份 base 版 bin,提取 finetune 版"
echo "=========================================="
date
# 备份 base bin 到独立目录
BACKUP_DIR=/home/zxh/projects/Kronos/finetune/feature_extraction/output/bins_base
mkdir -p $BACKUP_DIR
echo "备份 base bin → $BACKUP_DIR"
cd /home/zxh/projects/3.qlib_ifind_beta/data/qlib_root/features
for f in $(find . -name "kronos_*.day.bin" 2>/dev/null); do
    dest="$BACKUP_DIR/$(dirname $f)"
    mkdir -p "$dest"
    cp "$f" "$dest/"
done
echo "备份完成: $(find $BACKUP_DIR -name 'kronos_*.day.bin' | wc -l) 个 bin"

echo ""
echo "=========================================="
echo "阶段 4:finetune 版特征提取(6 GPU 并行)"
echo "=========================================="
date
cd /home/zxh/projects/Kronos/finetune/feature_extraction
$PY extract_parallel.py --checkpoint finetune --gpus 1,2,3,4,5,6

echo ""
echo "=========================================="
echo "阶段 5:LGBM-C(finetune 特征)"
echo "=========================================="
date
cd $LGBM_DIR
$LGBM_PY qrun/run.py qrun/workflow_kronos_ablation_C_finetune.yaml 2>&1 | tee /tmp/ablation_C.log

echo ""
echo "=========================================="
echo "✅ 全流程完成"
echo "=========================================="
date
echo "结果汇总:"
echo "  A baseline: IC=0.0548 RankIC=0.0612"
echo "  B base:     见 /tmp/ablation_B.log"
echo "  C finetune: 见 /tmp/ablation_C.log"
