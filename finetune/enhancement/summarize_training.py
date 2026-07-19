"""汇总日频 Loop 训练结果:各折 Val Loss + 收敛曲线 + 与 5min 对比。

训练完成后运行,生成训练报告(对应 loop-stage1-daily.md 的 ④ 度量评估)。

用法:
    cd /home/zxh/projects/Kronos/finetune
    PYTHONPATH=. python enhancement/summarize_training.py
"""
import os
import sys
import json
import glob
from pathlib import Path

ENH_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ENH_DIR.parent
SAVE_DIR = FINETUNE_DIR / "outputs" / "enhancement_daily"


def load_fold_summary(fold):
    """读某折的 summary.json(best_model 旁)"""
    fold_name = f"fold{fold}" if fold != 'full' else 'full'
    # summary.json 位置:train_predictor.py 保存的
    summary_path = SAVE_DIR / fold_name / "finetune_predictor" / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_val_loss_from_log(fold):
    """从训练日志提取每个 epoch 的 Val Loss(更详细)"""
    fold_name = f"fold{fold}" if fold != 'full' else 'foldfull'
    log_path = Path(f"/tmp/kronos_train/{fold_name}.log")
    if not log_path.exists():
        return []
    val_losses = []
    with open(log_path) as f:
        for line in f:
            if "Val Loss:" in line and "Best model" in line:
                # "Best model saved to ... (Val Loss: 2.2317)"
                try:
                    val = float(line.split("Val Loss:")[1].rstrip(")\n"))
                    val_losses.append(val)
                except (ValueError, IndexError):
                    pass
    return val_losses


def main():
    print("=" * 60)
    print("高贝塔指增 DAILY Loop - 训练结果汇总")
    print("=" * 60)

    folds = [0, 1, 2, 3, 'full']
    results = []

    print(f"\n{'Fold':<8} {'Val Loss':<12} {'Epochs':<10} {'训练时长':<12} {'状态'}")
    print("-" * 60)

    for fold in folds:
        summary = load_fold_summary(fold)
        if summary is None:
            print(f"{str(fold):<8} {'N/A':<12} {'N/A':<10} {'N/A':<12} ❌ 未训练/未找到")
            results.append({"fold": fold, "status": "missing"})
            continue

        # 从 summary 提取
        val_loss = summary.get("best_val_loss") or summary.get("val_loss")
        total_time = summary.get("total_time", "N/A")
        epochs = summary.get("epochs_completed", "N/A")

        # 从日志提取完整曲线
        curve = load_val_loss_from_log(fold)
        if curve and val_loss is None:
            val_loss = curve[-1]

        status = "✅" if val_loss is not None else "⚠️"
        print(f"{str(fold):<8} {str(val_loss):<12} {str(epochs):<10} {str(total_time):<12} {status}")

        results.append({
            "fold": fold,
            "val_loss": val_loss,
            "epochs": epochs,
            "total_time": total_time,
            "val_loss_curve": curve,
            "status": "ok",
        })

    # 汇总统计
    ok_results = [r for r in results if r["status"] == "ok" and r.get("val_loss") is not None]
    fold_results = [r for r in ok_results if r["fold"] != 'full']

    if fold_results:
        val_losses = [r["val_loss"] for r in fold_results]
        mean_vl = sum(val_losses) / len(val_losses)
        min_vl = min(val_losses)
        max_vl = max(val_losses)
        std_vl = (sum((v - mean_vl) ** 2 for v in val_losses) / len(val_losses)) ** 0.5

        print(f"\n{'='*60}")
        print(f"4 折统计(不含 full):")
        print(f"  平均 Val Loss: {mean_vl:.4f}")
        print(f"  标准差: {std_vl:.4f}")
        print(f"  最优折: fold{val_losses.index(min_vl)} ({min_vl:.4f})")
        print(f"  最差折: fold{val_losses.index(max_vl)} ({max_vl:.4f})")

        # 上次 5min 对比
        print(f"\n{'='*60}")
        print(f"与上次 5min 实验对比:")
        print(f"  {'方案':<20} {'平均 Val Loss':<15} {'最优单折':<15}")
        print(f"  {'-'*50}")
        print(f"  {'5min(上次)':<20} {'2.2159':<15} {'2.1887':<15}")
        print(f"  {'日频(本次)':<20} {mean_vl:<15.4f} {min_vl:<15.4f}")
        print(f"\n  注:Val Loss 是 token 重建误差,跨频率不可直接比。")
        print(f"     日频 K 线形态更多样(1年 vs 5天),重建难度本就更高。")
        print(f"     真正对比看回测的【超额 RankIC】。")

    # 保存汇总
    out_path = SAVE_DIR / "TRAINING_REPORT.json"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "folds": results,
            "mean_val_loss": mean_vl if fold_results else None,
            "std_val_loss": std_vl if fold_results else None,
        }, f, indent=2, default=str)
    print(f"\n汇总保存: {out_path}")
    print(f"\n下一步:回测 + 门禁检查")
    print(f"  cd finetune && PYTHONPATH=. python enhancement/backtest_excess.py --all")


if __name__ == "__main__":
    main()
