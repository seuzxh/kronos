import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalize the window. Mean and std are calculated strictly on the
        # lookback window (past data) to prevent future data leakage.
        past_len = self.config.lookback_window
        past_x = x[:past_len]

        x_mean = np.mean(past_x, axis=0)
        x_std  = np.std(past_x, axis=0)

        # Apply normalization and robust clipping to the entire sequence
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


class HighbetaDataset(Dataset):
    """
    高贝塔股池 5min K线数据集。

    与 QlibDataset 的差异:
    1. 数据源: 从 config_highbeta.py 加载配置, 读 fold{N}/{train,val}_5min.pkl
    2. 频率: 5min (由1min聚合而来), 时间特征仍从 datetime 派生
    3. 切分: 按 CV fold (环境变量 KRONOS_FOLD 控制), 而非固定时间区间
    4. 归一化: 与 QlibDataset 一致, 仅用 lookback 段算 mean/std (防未来泄露)

    环境变量:
      KRONOS_FOLD: fold 编号 (0,1,2,3) 或 'full', 决定加载哪个目录的 pickle

    Review 注意:
    - 跨日窗口: pickle中每股票序列保留跨日(隔夜缺口),窗口切片天然跨日
    - 归一化: x_mean/x_std 只在 x[:lookback] 段算, 完整窗口用该统计量归一化
    - 时间特征: 5min datetime 的 minute(0,5,...,55), hour(9,10,...,15), weekday 等
    """
    def __init__(self, data_type='train', fold=None):
        from config_highbeta import Config
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # fold: 优先用参数,其次环境变量,默认0
        import os
        if fold is None:
            fold = os.environ.get('KRONOS_FOLD', '0')
        self.fold = fold

        # 专用 RNG (避免干扰模型初始化)
        self.py_rng = random.Random(self.config.seed)

        # 路径和采样数
        fold_dir = self.config.get_cv_fold_dir(fold)
        if data_type == 'train':
            self.data_path = f"{fold_dir}/train_5min.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{fold_dir}/val_5min.pkl"
            self.n_samples = self.config.n_val_iter

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"数据文件不存在: {self.data_path}\n"
                f"请先运行 preprocess_5min.py 生成数据。"
            )

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # 预计算所有合法 (symbol, start_index) 对
        self.indices = []
        print(f"[{data_type.upper()}/fold{fold}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            # 兼容 index 无名的情况: reset_index 后第一列是时间戳
            # 统一重命名为 'datetime' (QlibDataset 约定)
            if 'datetime' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'datetime'})
            series_len = len(df)
            num_samples = series_len - self.window + 1
            if num_samples > 0:
                # 生成时间特征 (从5min datetime派生)
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # 只保留必要列节省内存
                self.data[symbol] = df[self.feature_list + self.time_feature_list]
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}/fold{fold}] Found {len(self.indices)} possible samples. "
              f"Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch):
        """每epoch重置采样RNG,保证分布式可复现。"""
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 从索引池随机采样 (idx 被忽略, 保证 DistributedSampler 控制顺序)
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # 归一化: 仅用 lookback 段算 mean/std (防未来泄露)
        past_len = self.config.lookback_window
        past_x = x[:past_len]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    import os
    # 根据环境变量决定用哪个数据集验证
    use_highbeta = os.environ.get('KRONOS_CONFIG') == 'config_highbeta'
    if use_highbeta:
        fold = os.environ.get('KRONOS_FOLD', '0')
        print(f"验证 HighbetaDataset (fold={fold})...")
        ds = HighbetaDataset(data_type='train', fold=fold)
        print(f"Dataset length: {len(ds)}")
        if len(ds) > 0:
            x, x_stamp = ds[0]
            print(f"Sample feature shape: {x.shape} (期望: ({ds.window}, 6))")
            print(f"Sample time shape: {x_stamp.shape} (期望: ({ds.window}, 5))")
            print(f"Feature range: [{x.min():.3f}, {x.max():.3f}] (clip={ds.config.clip})")
    else:
        # Example usage and verification.
        print("Creating training dataset instance...")
        train_dataset = QlibDataset(data_type='train')
        print(f"Dataset length: {len(train_dataset)}")
        if len(train_dataset) > 0:
            try_x, try_x_stamp = train_dataset[100]
            print(f"Sample feature shape: {try_x.shape}")
            print(f"Sample time feature shape: {try_x_stamp.shape}")
        else:
            print("Dataset is empty.")
