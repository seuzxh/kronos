"""
Configuration for high-beta universe (883926.TI) 5min Kronos fine-tuning.

核心改动 vs 原 finetune/config.py:
1. 数据源: qlib_data_path_1min = /home/zxh/cn_data_1min (1min bin)
2. 频率: 5min (由 1min 聚合,240根/天 → 48根/天)
3. 股池: 严格用 universe_snapshots.csv 的高贝塔股池(每日100只,日频换手)
4. 窗口: lookback=240(5交易日) → predict=48(次日全天),允许跨日
5. 切分: 4折扩张式时间序列CV + 最终全量模型(替代原固定3段)

Review 注意:
- 5min 聚合逻辑在 preprocess_5min.py,resample('5min').agg(...)
- CV 切分边界计算在 preprocess_5min.py,_compute_cv_splits()
- 归一化在 dataset.py,仅用 lookback 段(防泄露),与本配置无关
"""
import os


class Config:
    def __init__(self):
        # =================================================================
        # 数据源 & 股池
        # =================================================================
        # 1min bin provider (5min K线由此聚合)
        self.qlib_data_path_1min = "/home/zxh/cn_data_1min"
        # 日K provider (仅用于读取交易日历 day.txt,做 CV 切分)
        self.qlib_data_path_day = "/home/zxh/qlib_data"

        # 高贝塔股池快照 (date, code_ifind, code_qlib, name),609日×100只
        self.universe_csv = "/home/zxh/projects/3.qlib_ifind_beta/data/universe_snapshots.csv"

        # 1min 黑名单: 这4只 factor 退化成1.0,数据口径错误,必须排除
        # 来源: /home/zxh/cn_data_1min/instruments/blacklist_1min.txt
        self.blacklist = ["SH600599", "SH600636", "SH600696", "SH688287"]

        # 数据整体时间范围 (1min 真实数据从 2024-01-02 起)
        self.dataset_begin_time = "2024-01-02"
        self.dataset_end_time = "2026-07-10"   # universe_snapshots.csv 最后日期

        # =================================================================
        # 窗口参数 (5min K线)
        # =================================================================
        # lookback=240 = 5个交易日 × 48根/天; predict=48 = 次日全天
        # 训练/推理结构一致: 用前5交易日全天 + 预测次日全天
        self.lookback_window = 240
        self.predict_window = 48
        self.max_context = 512      # Kronos-base 上下文上限
        # 样本总长 = lookback + predict + 1 (next-token 平移用)
        # → 240 + 48 + 1 = 289

        # 特征列 (与 finetune/config.py 保持一致,6维)
        # qlib bin 无原生 amount,用 amt = vwap * volume 派生
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # 时间特征 (从 5min datetime 派生,5维)
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # 时间序列 CV 切分 (4折扩张式)
        # =================================================================
        self.cv_folds = 4
        self.cv_val_days = 20          # 每折验证段 20 交易日
        # 4折验证段共 80 日,从数据末尾倒推,训练段扩张式递增
        # 详细边界由 preprocess_5min.py 基于交易日历计算

        # =================================================================
        # 训练超参
        # =================================================================
        self.clip = 5.0                # z-score 后截断,防异常值

        self.epochs = 8
        self.log_interval = 100
        # 5min 序列长(289), 但模型仅102M, 实测显存:
        # batch=8→2.54GB, batch=16→4.58GB, batch=32→8.76GB (每卡)
        # sglang 占84GB/卡, 剩余~13GB, batch=32 安全
        self.batch_size = 32

        # 每 epoch 采样数 (按 batch 倍数定义,非全量遍历)
        self.n_train_iter = 2000 * self.batch_size   # 64000 样本/epoch
        self.n_val_iter = 400 * self.batch_size      # 12800 样本/epoch

        self.num_workers = 8

        # AMP (BF16, H20 原生支持)
        self.use_amp = True
        self.amp_dtype = "bfloat16"

        # 学习率
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 3e-5

        self.accumulation_steps = 4    # 有效 batch = 16 * 4GPU * 4 = 256

        # AdamW
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.05

        self.seed = 42

        # =================================================================
        # 路径
        # =================================================================
        # 数据预处理产物目录 (按 fold 组织)
        self.dataset_path = "./data/highbeta_5min"

        # 模型保存
        self.save_path = "./outputs/highbeta_5min"
        self.tokenizer_save_folder_name = 'finetune_tokenizer'
        self.predictor_save_folder_name = 'finetune_predictor'

        # =================================================================
        # 预训练模型 (从 HF 缓存加载,无需联网)
        # 已确认缓存:
        #   /home/zxh/.cache/huggingface/hub/models--NeoQuasar--Kronos-Tokenizer-base
        #   /home/zxh/.cache/huggingface/hub/models--NeoQuasar--Kronos-base
        # =================================================================
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 强制离线,避免联网卡住
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"

        # 微调产物路径 (派生)
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
            "project_name": "Kronos-Highbeta-5min",
            "workspace": "local",
        }
        self.comet_tag = 'highbeta_5min'
        self.comet_name = 'highbeta_5min'

        # =================================================================
        # 推理参数 (qlib_test.py 用)
        # =================================================================
        self.inference_T = 1.0          # 采样温度,1.0 = 接近贪心
        self.inference_top_p = 0.95
        self.inference_top_k = 0
        self.inference_sample_count = 20  # 多次采样取均值,降噪

    def get_cv_fold_dir(self, fold):
        """返回某折的数据目录路径。fold ∈ {0,1,2,3,'full'}."""
        return f"{self.dataset_path}/fold{fold}"

    def get_cv_save_dir(self, fold):
        """返回某折的模型保存目录。fold ∈ {0,1,2,3,'full'}."""
        return f"{self.save_path}/fold{fold}"


if __name__ == '__main__':
    # 配置自检: 打印关键参数
    c = Config()
    print("=" * 60)
    print("高贝塔 5min 微调配置")
    print("=" * 60)
    print(f"数据源(1min): {c.qlib_data_path_1min}")
    print(f"股池: {c.universe_csv}")
    print(f"黑名单: {c.blacklist}")
    print(f"数据范围: {c.dataset_begin_time} → {c.dataset_end_time}")
    print(f"窗口: lookback={c.lookback_window} predict={c.predict_window} "
          f"总长={c.lookback_window + c.predict_window + 1}")
    print(f"CV: {c.cv_folds}折 × 验证{c.cv_val_days}日")
    print(f"batch={c.batch_size} epochs={c.epochs} "
          f"n_train_iter={c.n_train_iter} lr={c.predictor_learning_rate}")
    print(f"预训练: {c.pretrained_predictor_path}")
    print(f"数据产物: {c.dataset_path}")
    print(f"模型产物: {c.save_path}")
