"""
高贝塔指增 DAILY 数据集。

设计决策(见 design.md §3.3):
Stage 1 先跑"无指数特征 baseline"(H1 对照组),即:
- Kronos 只吃个股 6 维 OHLCV(原生 d_in=6,不改 tokenizer 架构)
- 指数特征仅在"回测信号层"使用(预测个股收益 − 实际指数收益 = 超额收益)
- 这样 H1 假设(指数特征是否有用)才能被干净对比

后续 H1 通过后,再扩展为 12 维(加 idx_* 特征)——那时改 dataset 即可,
模型架构不动(用方案 a:两路 6→d_model 映射后相加,需小改 tokenizer 第一层)。

因此本类目前与 HighbetaDataset 同构,只是数据源换成日频 pickle。
"""
import os
import sys
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ENH_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ENH_DIR.parent
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))


class EnhancementDailyDataset(Dataset):
    """高贝塔指增日频数据集(Stage 1 baseline,个股 6 维)。

    数据源:fold{N}/{train,val}_daily.pkl(dict[symbol -> DataFrame[12 cols]])
    当前只用前 6 列(个股 OHLCV),后 6 列(指数)留给回测信号层。

    归一化:与 HighbetaDataset 一致,仅用 lookback 段算 mean/std(防未来泄露)
    """

    def __init__(self, data_type='train', fold=None, config=None):
        # config 参数仅为接口兼容,实际用 Config 实例(与 HighbetaDataset 风格一致)
        from enhancement.config_daily import Config
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        if fold is None:
            fold = os.environ.get('KRONOS_FOLD', '0')
        self.fold = fold

        self.py_rng = random.Random(self.config.seed)

        fold_dir = self.config.get_cv_fold_dir(fold)
        if data_type == 'train':
            self.data_path = f"{fold_dir}/train_daily.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{fold_dir}/val_daily.pkl"
            self.n_samples = self.config.n_val_iter

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"数据文件不存在: {self.data_path}\n"
                f"请先运行 enhancement/preprocess_daily.py 生成数据。"
            )

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1  # 246
        self.symbols = list(self.data.keys())
        # 个股 6 维特征(原生 Kronos d_in)
        self.feature_list = self.config.feature_list
        # 指数 6 维(预加载但训练不用,留作未来扩展)
        self.index_feature_list = self.config.index_feature_list
        self.time_feature_list = self.config.time_feature_list

        # 预计算所有合法 (symbol, start_index) 对
        # 对 val 集:predict 段(末尾 predict_window+1 根)必须落在验证段(序列末尾 20 日)
        # 这是防数据泄露的关键 —— lookback 用历史允许,但 predict 必须在验证段内
        cv_val_days = getattr(self.config, 'cv_val_days', 20)
        lookback = self.config.lookback_window

        self.indices = []
        print(f"[{data_type.upper()}/fold{fold}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            if 'datetime' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'datetime'})
            df['datetime'] = pd_datetime(df['datetime'])
            series_len = len(df)
            num_samples = series_len - self.window + 1
            if num_samples > 0:
                df['minute'] = 0
                df['hour'] = 0
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                cols_to_keep = self.feature_list + self.time_feature_list
                self.data[symbol] = df[cols_to_keep]
                for i in range(num_samples):
                    # val 集限制:predict 段末尾(start_idx + window - 1)
                    # 必须落在序列末尾 cv_val_days 日内(验证段)
                    if data_type == 'val':
                        predict_end_idx = i + self.window - 1  # window 最后一根
                        if predict_end_idx < series_len - cv_val_days:
                            continue  # predict 不在验证段,跳过
                    self.indices.append((symbol, i))

        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}/fold{fold}] Found {len(self.indices)} possible samples. "
              f"Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch):
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # 归一化:仅用 lookback 段算 mean/std(防未来泄露)
        past_len = self.config.lookback_window
        past_x = x[:past_len]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


def pd_datetime(series):
    """安全转 datetime(兼容 string/datetime 混合)"""
    import pandas as pd
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series)


if __name__ == '__main__':
    # 自检
    fold = os.environ.get('KRONOS_FOLD', '0')
    print(f"验证 EnhancementDailyDataset (fold={fold})...")
    ds = EnhancementDailyDataset(data_type='train', fold=fold)
    x, x_stamp = ds[0]
    print(f"Sample feature shape: {x.shape}  (期望 [{ds.window}, 6])")
    print(f"Sample time shape: {x_stamp.shape}  (期望 [{ds.window}, 5])")
    print(f"Feature range: [{x.min():.3f}, {x.max():.3f}] (clip={ds.config.clip})")
