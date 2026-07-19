"""
高贝塔指增 DAILY 训练管线 Smoke Test(轻量验证,不跑完整 DDP)。

验证内容:
1. EnhancementDailyDataset 能加载 fold0 数据
2. 预训练 Kronos-Tokenizer / Kronos 能从 HF 缓存加载
3. 取 1 个 batch,tokenizer.encode + model.forward + loss 能跑通
4. 输出各步 tensor shape,确认维度正确(246, 6)

用法:
  cd /home/zxh/projects/Kronos/finetune
  KRONOS_CONFIG=config_daily KRONOS_FOLD=0 python enhancement/smoke_daily.py
"""
import os
import sys
import torch
from pathlib import Path

ENH_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = ENH_DIR.parent
PROJECT_ROOT = FINETUNE_DIR.parent
sys.path.insert(0, str(FINETUNE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('KRONOS_CONFIG', 'config_daily')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from enhancement.config_daily import Config
from enhancement.dataset_enh import EnhancementDailyDataset
from model.kronos import KronosTokenizer, Kronos


def main():
    cfg = Config()
    fold = os.environ.get('KRONOS_FOLD', '0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}, fold={fold}")

    # 1. 加载数据集
    print("\n[1] 加载 EnhancementDailyDataset...")
    ds = EnhancementDailyDataset('train', fold=fold)
    print(f"  数据集大小: {len(ds)} 样本/epoch")
    print(f"  window: {ds.window} (期望: lookback+predict+1 = {cfg.lookback_window + cfg.predict_window + 1})")

    # 2. 加载预训练模型
    print("\n[2] 加载预训练模型...")
    tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path)
    tokenizer.eval().to(device)
    print(f"  Tokenizer 加载成功 (d_in={tokenizer.d_in}, 期望 6)")

    model = Kronos.from_pretrained(cfg.pretrained_predictor_path)
    model.to(device).eval()
    print(f"  Kronos predictor 加载成功")

    # 3. 取 1 个 batch 测试前向
    print("\n[3] 取 1 个 batch 测试前向...")
    x, x_stamp = ds[0]
    x = x.unsqueeze(0).to(device)
    x_stamp = x_stamp.unsqueeze(0).to(device)
    print(f"  输入 x shape: {x.shape} (期望: (1, {ds.window}, 6))")
    print(f"  输入 x_stamp shape: {x_stamp.shape} (期望: (1, {ds.window}, 5))")

    # 4. tokenizer encode
    with torch.no_grad():
        token_s1, token_s2 = tokenizer.encode(x, half=True)
        print(f"  tokenize s1 shape: {token_s1.shape} (期望: (1, {ds.window}))")
        print(f"  tokenize s2 shape: {token_s2.shape}")

        # 5. next-token 构造 + model forward
        token_in_s1 = token_s1[:, :-1]
        token_in_s2 = token_s2[:, :-1]
        stamp_in = x_stamp[:, :-1, :]
        target_s1 = token_s1[:, 1:]
        target_s2 = token_s2[:, 1:]

        logits_s1, logits_s2 = model(token_in_s1, token_in_s2, stamp_in)
        print(f"  logits_s1 shape: {logits_s1.shape}")
        print(f"  logits_s2 shape: {logits_s2.shape}")

        # 6. loss
        loss, s1_loss, s2_loss = model.head.compute_loss(
            logits_s1, logits_s2, target_s1, target_s2
        )
        print(f"  loss: {loss.item():.4f} (s1: {s1_loss.item():.4f}, s2: {s2_loss.item():.4f})")

    print("\n✅ Smoke test 通过! 指增日频训练管线可正常运行。")
    print(f"   下一步: bash enhancement/run_daily.sh train 0")


if __name__ == '__main__':
    main()
