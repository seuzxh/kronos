"""
Configuration for high-beta enhancement DAILY Kronos fine-tuning.

指增方案日频 Loop 配置(Stage 1)。

与 config_highbeta.py(5min)的差异:
1. 频率: 日线(直接读 qlib day.bin),非 5min 聚合
2. 数据源: qlib_data_path_day 直接读日线 + index_883926.csv 读指数日线
3. 特征维度: d_in=12(个股6 + 指数6),非 6
4. 窗口: lookback=240 交易日(~1年) → predict=5 交易日,总长 246
5. 预测目标: 仍为 K 线 token(原生),但回测信号用"预测超额收益"

Review 注意:
- 指数特征作为"条件输入",与个股特征拼接为 12 维
- KronosTokenizer d_in=6 不能直接吃 12 维;dataset 把 12 维拆成两路各自归一化,
  实际喂给 tokenizer 时只在个股 6 维上做 token 化(指数 6 维仅作时间特征式的条件注入)
  → 见 dataset_enh.py 的 _inject_index_condition 实现
"""
import os
import sys
from pathlib import Path

# 让 config 能被 train_predictor.py 等通过 KRONOS_CONFIG 环境变量导入
ENH_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ENH_DIR.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))


class Config:
    def __init__(self):
        # =================================================================
        # 数据源 & 股池
        # =================================================================
        self.qlib_data_path_day = "/home/zxh/qlib_local_data/cn_data"

        # 高贝塔成分股快照(由 fetch_universe.py 拉取)
        self.universe_csv = str(ENH_DIR.parent / "data" / "enhancement" / "universe_highbeta.csv")

        # 高贝塔指数日线(由 fetch_index_daily.py 拉取)
        self.index_csv = str(ENH_DIR.parent / "data" / "enhancement" / "index_883926.csv")
        self.index_symbol = "883926.TI"

        # 数据整体时间范围(与 universe 快照对齐)
        self.dataset_begin_time = "2024-01-02"
        self.dataset_end_time = "2026-07-17"

        # =================================================================
        # 窗口参数(日线)
        # =================================================================
        # lookback=240 ≈ 1 年交易日;predict=5 = 预测未来一周
        self.lookback_window = 240
        self.predict_window = 5
        self.max_context = 512
        # 总长 = 240 + 5 + 1 = 246 < 512 ✅

        # 特征列(个股 6 维,与 config_highbeta.py 一致)
        # qlib 日线 bin 字段:open/high/low/close/volume/amount
        # 统一成 Kronos 约定:open/high/low/close/vol/amt
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']

        # 指数特征列(同 6 维,从 index_csv 派生)
        # 与个股对齐时间轴,做相同窗口 z-score 归一化
        self.index_feature_list = ['idx_open', 'idx_high', 'idx_low', 'idx_close', 'idx_vol', 'idx_amt']

        # 时间特征(日线: weekday/day/month,minute/hour 置 0 占位以兼容 Kronos)
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # CV 切分(4 折扩张式)
        # =================================================================
        # 日线数据 ~600 个交易日,4 折 × 验证 20 日 = 80 日,扩张式
        self.cv_folds = 4
        self.cv_val_days = 20

        # =================================================================
        # 训练超参
        # =================================================================
        self.clip = 5.0

        self.epochs = 8
        self.log_interval = 100
        # 日线窗口短(246),显存占用比 5min(289)小,batch 可更大
        self.batch_size = 32

        # 日频样本比 5min 少(日线根数少),多采样
        self.n_train_iter = 2000 * self.batch_size   # 64000 样本/epoch
        self.n_val_iter = 400 * self.batch_size

        self.num_workers = 8
        self.use_amp = True
        self.amp_dtype = "bfloat16"

        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 3e-5
        self.accumulation_steps = 4

        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.05

        self.seed = 42

        # =================================================================
        # 路径
        # =================================================================
        self.dataset_path = str(FINETUNE_DIR / "data" / "enhancement_daily")
        self.save_path = str(FINETUNE_DIR / "outputs" / "enhancement_daily")
        self.tokenizer_save_folder_name = 'finetune_tokenizer'
        self.predictor_save_folder_name = 'finetune_predictor'

        # =================================================================
        # 预训练模型(与 config_highbeta.py 一致,从 HF 缓存离线加载)
        # =================================================================
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"

        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )

        # =================================================================
        # 实验控制
        # =================================================================
        self.use_comet = False
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Enhancement-Daily",
            "workspace": "local",
        }
        self.comet_tag = 'enhancement_daily'
        self.comet_name = 'enhancement_daily'

        # =================================================================
        # 推理参数
        # =================================================================
        self.inference_T = 1.0
        self.inference_top_p = 0.95
        self.inference_top_k = 0
        self.inference_sample_count = 20

        # =================================================================
        # 回测参数(指增专用)
        # =================================================================
        self.backtest_top_k = 10           # 选股数(可调整,见 loop-stage1 §6)
        self.backtest_cost_bps = 10        # 单边手续费 0.1%(10bps)
        self.rankic_gate_threshold = 0.03  # Stage1→Stage2 门禁阈值

    def get_cv_fold_dir(self, fold):
        """fold ∈ {0,1,2,3,'full'}"""
        suffix = str(fold) if fold == 'full' else f"fold{fold}"
        return f"{self.dataset_path}/{suffix}"

    def get_cv_save_dir(self, fold):
        suffix = str(fold) if fold == 'full' else f"fold{fold}"
        return f"{self.save_path}/{suffix}"


if __name__ == '__main__':
    c = Config()
    print("=" * 60)
    print("高贝塔指增 DAILY 微调配置 (Stage 1)")
    print("=" * 60)
    print(f"数据源(日线): {c.qlib_data_path_day}")
    print(f"股池: {c.universe_csv}")
    print(f"指数: {c.index_csv} ({c.index_symbol})")
    print(f"数据范围: {c.dataset_begin_time} → {c.dataset_end_time}")
    print(f"窗口: lookback={c.lookback_window} predict={c.predict_window} "
          f"总长={c.lookback_window + c.predict_window + 1}")
    print(f"特征: 个股{len(c.feature_list)}维 + 指数{len(c.index_feature_list)}维")
    print(f"CV: {c.cv_folds}折 × 验证{c.cv_val_days}日")
    print(f"batch={c.batch_size} epochs={c.epochs} "
          f"n_train_iter={c.n_train_iter} lr={c.predictor_learning_rate}")
    print(f"预训练: {c.pretrained_predictor_path}")
    print(f"数据产物: {c.dataset_path}")
    print(f"模型产物: {c.save_path}")
    print(f"门禁阈值: 超额 RankIC > {c.rankic_gate_threshold}")
