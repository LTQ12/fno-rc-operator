import os
import h5py
import torch
import numpy as np
import argparse
import torch.nn.functional as F
from timeit import default_timer

from utilities3 import LpLoss, UnitGaussianNormalizer
from fourier_3d_unet import U_FNO_3d


def build_windows(series, T_in, T_out, stride, max_windows_per_sample=None):
    """series: (N,S1,S2,T_total) -> X:(M,S1,S2,T_in), Y:(M,S1,S2,T_out), starts(M,), T_total(int)"""
    N, S1, S2, T_total = series.shape
    xs, ys, starts = [], [], []
    for n in range(N):
        positions = list(range(0, max(1, T_total - (T_in + T_out) + 1), stride))
        if max_windows_per_sample is not None:
            positions = positions[:max_windows_per_sample]
        for t0 in positions:
            xs.append(series[n, :, :, t0:t0+T_in])
            ys.append(series[n, :, :, t0+T_in:t0+T_in+T_out])
            starts.append(t0)
    X = torch.from_numpy(np.stack(xs, 0)).float()   # (M,S1,S2,T_in)
    Y = torch.from_numpy(np.stack(ys, 0)).float()   # (M,S1,S2,T_out)
    starts = torch.tensor(starts, dtype=torch.float32)
    return X, Y, starts, T_total


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 读取数据 (S1,S2,T,N) -> (N,S1,S2,T)
    with h5py.File(args.data_path, 'r') as f:
        u_np = f['u'][()]
    u_np = np.transpose(u_np, (3, 0, 1, 2))

    ntrain, ntest = args.ntrain, args.ntest
    train_series = u_np[:ntrain]
    test_series  = u_np[-ntest:]

    # 顺序分块：步长 T_out（与其他模型一致）
    train_x, train_y, train_starts, train_T = build_windows(train_series, args.T_in, args.T_out, stride=args.T_out)
    test_x,  test_y,  test_starts,  test_T  = build_windows(test_series,  args.T_in, args.T_out, stride=args.T_out)

    print(f"Data prepared. Train x: {train_x.shape}, Train y: {train_y.shape}")

    # 目标归一化（主指标在归一化空间）
    y_normalizer = UnitGaussianNormalizer(train_y)

    # 绝对时间通道
    def append_abs_time(x, starts, T_total, T_in):
        num, S1, S2 = x.shape[0], x.shape[1], x.shape[2]
        t_abs = (starts / float(T_total)).view(num, 1, 1, 1).expand(num, S1, S2, 1)
        return torch.cat([x, t_abs], dim=-1)  # (num,S1,S2,T_in+1)

    train_x = append_abs_time(train_x, train_starts, train_T, args.T_in)
    test_x = append_abs_time(test_x, test_starts, test_T, args.T_in)

    # 构造 U-FNO 输入：(B,S1,S2,1,T_in+1) → repeat 到 D=T_out
    S1, S2 = train_x.shape[1], train_x.shape[2]
    train_x = train_x.reshape(train_x.shape[0], S1, S2, 1, args.T_in+1).repeat(1, 1, 1, args.T_out, 1)
    test_x  = test_x.reshape(test_x.shape[0],  S1, S2, 1, args.T_in+1).repeat(1, 1, 1, args.T_out, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x,  test_y),  batch_size=args.batch_size, shuffle=False)

    # 模型：in_channels=T_in+1, out_channels=1（与 FNO/FNO-RC 保持一致）
    model = U_FNO_3d(args.modes, args.modes, args.modes, args.width, in_channels=args.T_in+1, out_channels=1).to(device)
    print(f"Training U-FNO-3D with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    myloss = LpLoss(size_average=False)
    y_normalizer.to(device)

    for ep in range(args.epochs):
        model.train()
        train_l2 = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)  # (B,S1,S2,T_out)
            loss = myloss(y_normalizer.encode(out).view(x.shape[0], -1), y_normalizer.encode(y).view(x.shape[0], -1))
            loss.backward(); optimizer.step()
            train_l2 += loss.item()
        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze(-1)
                test_l2 += myloss(y_normalizer.encode(out).view(x.shape[0], -1), y_normalizer.encode(y).view(x.shape[0], -1)).item()

        train_l2 /= train_x.shape[0]
        test_l2  /= test_x.shape[0]
        if (ep + 1) % 10 == 0:
            print(f"Epoch: {ep+1} | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")

    # 保存模型与 normalizer
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_save_path)
    yn_stats = {'mean': y_normalizer.mean.detach().cpu(), 'std': y_normalizer.std.detach().cpu()}
    torch.save(yn_stats, os.path.join(os.path.dirname(args.model_save_path), 'u_fno_y_normalizer.pt'))
    print(f"Model saved to {args.model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-FNO-3D with unified protocol.')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--model_save_path', type=str, default='/content/models/u_fno_3d.pt')
    parser.add_argument('--ntrain', type=int, default=40)
    parser.add_argument('--ntest', type=int, default=10)
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)


