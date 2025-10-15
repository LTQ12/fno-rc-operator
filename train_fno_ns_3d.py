import torch
import numpy as np
import torch.nn as nn
import h5py
import torch.nn.functional as F
import os

from utilities3 import LpLoss, save_checkpoint, UnitGaussianNormalizer
from fourier_3d_clean import FNO3d # Corrected class name

# ##############################################################################
# This script trains the standard FNO model on the 3D Navier-Stokes dataset.
# It serves as the baseline for comparing against experimental models.
# ##############################################################################

def main():
    # ################################
    # Configurations
    # ################################
    # Path for Google Colab execution from mounted Google Drive
    TRAIN_PATH = '/content/data/ns_V1e-4_N10000_T30.mat'
    # Model path, saved to Google Drive for persistence
    MODEL_PATH = '/content/drive/MyDrive/fno_models/fno_3d_standard.pt'
    
    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Dataset and Loader parameters (align with RC sliding windows)
    ntrain_base = 40
    ntest_base = 10
    T_in = 10
    T_out = 20 # 多步预测
    # 顺序块训练（TBPTT式）：按时间顺序分块，步长取 T_out，避免随机滑窗分布漂移
    stride = T_out
    stride_eval = T_out
    max_windows_per_sample = 25
    max_windows_per_sample_eval = 5

    # Model parameters
    modes = 8
    width = 20

    # Training parameters
    batch_size = 10
    learning_rate = 0.001
    epochs = 50 # A reasonable number for a first run
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ################################
    # Data Loading and Preparation
    # ################################
    print("Loading data...")
    with h5py.File(TRAIN_PATH, 'r') as f:
        all_data = torch.from_numpy(f['u'][()]).float() # (S,S,T,N)
        all_data = all_data.permute(3, 1, 2, 0) # (N,S,S,T)

    # Split base sequences
    train_series = all_data[:ntrain_base]
    test_series = all_data[-ntest_base:]

    # Build sliding windows identical to RC
    def build_windows(series, T_in, T_out, stride, max_windows_per_sample=None):
        B, S1, S2, T_total = series.shape
        xs, ys, starts_all = [], [], []
        for b in range(B):
            starts = list(range(0, max(1, T_total - (T_in + T_out) + 1), stride))
            if max_windows_per_sample is not None:
                starts = starts[:max_windows_per_sample]
            for t0 in starts:
                x = series[b, :, :, t0:t0+T_in]
                y = series[b, :, :, t0+T_in:t0+T_in+T_out]
                xs.append(x)
                ys.append(y)
                starts_all.append(t0)
        X = torch.stack(xs, dim=0)
        Y = torch.stack(ys, dim=0)
        starts_tensor = torch.tensor(starts_all, dtype=X.dtype)
        return X, Y, starts_tensor, T_total

    train_x, train_y, train_starts, train_T = build_windows(train_series, T_in, T_out, stride, max_windows_per_sample)
    test_x, test_y, test_starts, test_T = build_windows(test_series, T_in, T_out, stride_eval, max_windows_per_sample_eval)

    # 绝对时间通道（两模型统一）：t_abs = t0 / T_total，扩展为 (S,S,T_in) 并与输入拼接
    def append_abs_time(x, starts, T_total):
        num = x.shape[0]
        S1, S2 = x.shape[1], x.shape[2]
        # 仅添加一个常量通道，拼接后为 T_in+1
        t_abs = (starts / float(T_total)).view(num, 1, 1, 1).expand(num, S1, S2, 1)
        return torch.cat([x, t_abs], dim=-1)

    train_x = append_abs_time(train_x, train_starts, train_T)
    test_x = append_abs_time(test_x, test_starts, test_T)

    # Repeat input along D to length T_out, keep last dim as in_channels=T_in
    S = train_x.shape[1]
    train_x = train_x.reshape(train_x.shape[0], S, S, 1, T_in+1).repeat(1, 1, 1, T_out, 1)
    test_x = test_x.reshape(test_x.shape[0], S, S, 1, T_in+1).repeat(1, 1, 1, T_out, 1)
    
    print(f"Data prepared. Train x: {train_x.shape}, Train y: {train_y.shape}")

    # Normalizer for targets – ensure FNO uses the same (norm-space) metric as FNO-RC
    y_normalizer = UnitGaussianNormalizer(train_y)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False
    )
    
    # ################################
    # Model, Optimizer, and Loss
    # ################################
    model = FNO3d(
        modes1=modes, modes2=modes, modes3=modes,
        width=width,
        in_channels=T_in+1,
        out_channels=1
    ).to(device)
    
    print(f"Training Standard FNO with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    myloss = LpLoss(size_average=False)

    # ################################
    # Training Loop
    # ################################
    # Move normalizer to device for encode/decode ops
    y_normalizer.to(device)

    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            out_s = out.squeeze(-1)

            # Compute loss in normalized space (fair with FNO-RC)
            loss = myloss(
                y_normalizer.encode(out_s).view(x.shape[0], -1),
                y_normalizer.encode(y).view(x.shape[0], -1)
            )
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                out_s = out.squeeze(-1)
                # Evaluate primary metric in normalized space
                test_l2 += myloss(
                    y_normalizer.encode(out_s).view(x.shape[0], -1),
                    y_normalizer.encode(y).view(x.shape[0], -1)
                ).item()

        train_l2 /= train_x.shape[0]
        test_l2 /= test_x.shape[0]

        if (ep + 1) % 10 == 0:
            print(f"Epoch: {ep+1} | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")
        
    save_checkpoint(MODEL_PATH, model, optimizer)
    # Save target normalizer stats for consistent visualization
    yn_stats = {
        'mean': y_normalizer.mean.detach().cpu(),
        'std':  y_normalizer.std.detach().cpu()
    }
    yn_path = os.path.join(os.path.dirname(MODEL_PATH), 'fno_y_normalizer.pt')
    torch.save(yn_stats, yn_path)
    print(f"Training complete. Standard FNO model saved to {MODEL_PATH}\nSaved FNO y_normalizer stats to {yn_path}")

if __name__ == "__main__":
    main() 