import os
import h5py
import torch
import numpy as np
import argparse
from timeit import default_timer

from utilities3 import LpLoss, UnitGaussianNormalizer
from fourier_3d_afno import AFNO3D


def build_windows(series, T_in, T_out, stride):
    N, S1, S2, T_total = series.shape
    xs, ys, starts = [], [], []
    for n in range(N):
        for t0 in range(0, max(1, T_total - (T_in + T_out) + 1), stride):
            xs.append(series[n, :, :, t0:t0+T_in])
            ys.append(series[n, :, :, t0+T_in:t0+T_in+T_out])
            starts.append(t0)
    X = torch.from_numpy(np.stack(xs, 0)).float()
    Y = torch.from_numpy(np.stack(ys, 0)).float()
    starts = torch.tensor(starts, dtype=torch.float32)
    return X, Y, starts, T_total


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with h5py.File(args.data_path, 'r') as f:
        u_np = f['u'][()]
    u_np = np.transpose(u_np, (3, 0, 1, 2))

    train_series = u_np[:args.ntrain]
    test_series  = u_np[-args.ntest:]

    train_x, train_y, train_starts, train_T = build_windows(train_series, args.T_in, args.T_out, stride=args.T_out)
    test_x,  test_y,  test_starts,  test_T  = build_windows(test_series,  args.T_in, args.T_out, stride=args.T_out)

    print(f"Data prepared. Train x: {train_x.shape}, Train y: {train_y.shape}")

    y_normalizer = UnitGaussianNormalizer(train_y)
    y_normalizer.to(device)

    # 构造 AFNO 输入：(B,H,W,D,T_in+1)，与 FNO 一致（绝对时间通道）
    def append_abs_time(x, starts, T_total, T_in):
        num, S1, S2 = x.shape[0], x.shape[1], x.shape[2]
        t_abs = (starts / float(T_total)).view(num, 1, 1, 1).expand(num, S1, S2, 1)
        return torch.cat([x, t_abs], dim=-1)

    train_x = append_abs_time(train_x, train_starts, train_T, args.T_in)
    test_x  = append_abs_time(test_x,  test_starts,  test_T,  args.T_in)

    S1, S2 = train_x.shape[1], train_x.shape[2]
    train_x = train_x.reshape(train_x.shape[0], S1, S2, 1, args.T_in+1).repeat(1,1,1,args.T_out,1)
    test_x  = test_x.reshape(test_x.shape[0],  S1, S2, 1, args.T_in+1).repeat(1,1,1,args.T_out,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x,  test_y),  batch_size=args.batch_size, shuffle=False)

    model = AFNO3D(args.modes, args.modes, args.modes, width=args.width, in_channels=args.T_in+1, out_channels=1, depth=args.depth).to(device)
    print(f"Training AFNO-3D with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    myloss = LpLoss(size_average=False)

    for ep in range(args.epochs):
        model.train()
        train_l2 = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
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

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AFNO-3D (unified protocol).')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--model_save_path', type=str, default='/content/models/afno_3d.pt')
    parser.add_argument('--ntrain', type=int, default=40)
    parser.add_argument('--ntest', type=int, default=10)
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)


