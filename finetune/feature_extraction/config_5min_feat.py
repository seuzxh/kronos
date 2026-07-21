"""
5min 特征提取器配置。

与 config_highbeta.py 的关系:
- 复用 highbeta_5min/full 的微调 checkpoint(不重训)
- 复用预训练 base 做 ablation 对照
- 1min 数据从备份目录读(主目录 cn_data_1min 已清空)
- 输出到 LGBM qlib_root,对齐 close.day.bin

关键参数(与训练时严格一致,否则 checkpoint 失配):
  lookback_window = 240   (5 交易日 × 48 根 5min)
  predict_window  = 48    (次日全天 48 根 5min)
  d_in            = 6     (open/high/low/close/vol/amt)
  max_context     = 512
  clip            = 5.0
"""
import os
from pathlib import Path

FEAT_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = FEAT_DIR.parent
PROJECT_ROOT = FINETUNE_DIR.parent


class Config:
    def __init__(self):
        # =================================================================
        # 数据源
        # =================================================================
        # 1min bin 备份目录(主目录 /home/zxh/cn_data_1min 已清空,从备份读)
        # bak.20260720 与 bak.242 内容一致,优先用最新的
        self.data_1min_root = "/home/zxh/cn_data_1min.bak.20260720"

        # 日线 bin(LGBM qlib_root,写特征的目标位置)
        self.qlib_root_lgbm = "/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root"
        self.day_features_root = f"{self.qlib_root_lgbm}/features"

        # 高贝塔股池快照
        self.universe_csv = "/home/zxh/projects/3.qlib_ifind_beta/data/universe_snapshots.csv"

        # 时间范围(覆盖 LGBM 训练+验证+测试段)
        # LGBM: train 2024-01-01→2025-12-31, valid 2026-01→03, test 2026-04→07-02
        # 注意:1min 数据从 2024-01-02 起,lookback=240(5 交易日)
        #       所以决策日从 2024-01-09 开始(留 5 日历史 buffer)
        self.start_date = "2024-01-09"
        self.end_date = "2026-07-02"

        # =================================================================
        # Kronos 窗口参数(必须与训练时严格一致)
        # =================================================================
        self.lookback_window = 240   # 5 交易日 × 48 根 5min
        self.predict_window = 48     # 次日全天
        self.max_context = 512
        self.clip = 5.0
        self.d_in = 6

        # 特征列(与 config_highbeta.py 一致)
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']

        # =================================================================
        # 推理参数
        # =================================================================
        # sample_count=5:一次推理内并行 5 条采样路径
        # 用于算 kronos_uncertainty(路径间标准差),所以不能 <2
        self.sample_count = 5
        self.inference_T = 1.0
        self.inference_top_p = 0.95
        self.inference_top_k = 0

        # =================================================================
        # Checkpoint 路径(两套做 ablation)
        # =================================================================
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"

        # 预训练 base(zero-shot 对照)
        self.base_predictor_path = "NeoQuasar/Kronos-base"

        # 微调 full(在 highbeta 股池上微调过)
        self.finetune_predictor_path = (
            f"{PROJECT_ROOT}/finetune/outputs/highbeta_5min/full/"
            f"finetune_predictor/checkpoints/best_model"
        )

        # =================================================================
        # 8 个输出特征名(写入 .day.bin,对齐 close.day.bin)
        # =================================================================
        self.kronos_fields = [
            'kronos_ret_intraday',      # 预测 T 日累计收益(收/开-1)
            'kronos_high_ratio',        # 预测日内最高涨幅
            'kronos_low_ratio',         # 预测日内最大跌幅(负数)
            'kronos_range',             # 预测日内振幅 (high-low)/open
            'kronos_close_pos',         # 预测收盘相对位置 (收-低)/(高-低)
            'kronos_morning_ret',       # 预测上午收益 (13:00 vs 09:30)
            'kronos_afternoon_ret',     # 预测下午收益 (15:00 vs 13:00)
            'kronos_uncertainty',       # 5 次采样间收益标准差
        ]

        # =================================================================
        # 输出目录(特征中间结果,parquet 格式,便于检查)
        # =================================================================
        self.output_dir = f"{PROJECT_ROOT}/finetune/feature_extraction/output"

        # =================================================================
        # 运行控制
        # =================================================================
        self.device = "cuda"  # 单卡共存,sglang 占满但剩 ~14GB 够用

        # 断点续跑:已完成的 (symbol, date) 跳过
        self.resume = True


if __name__ == '__main__':
    c = Config()
    print("=" * 60)
    print("5min 特征提取器配置")
    print("=" * 60)
    print(f"1min 数据: {c.data_1min_root}")
    print(f"LGBM qlib_root: {c.qlib_root_lgbm}")
    print(f"股池: {c.universe_csv}")
    print(f"时间范围: {c.start_date} → {c.end_date}")
    print(f"窗口: lookback={c.lookback_window} predict={c.predict_window}")
    print(f"采样: sample_count={c.sample_count} T={c.inference_T} top_p={c.inference_top_p}")
    print(f"特征数: {len(c.kronos_fields)}")
    print(f"  {c.kronos_fields}")
    print(f"base checkpoint: {c.base_predictor_path}")
    print(f"finetune checkpoint: {c.finetune_predictor_path}")
