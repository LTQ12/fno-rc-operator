"""
Trainer for the 3D FNO with CFT-based Residual Correction (FNO-RC-3D).
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from timeit import default_timer
import os
import h5py

from fourier_3d_cft_residual import FNO_RC_3D
from utilities3 import LpLoss, count_params, MatReader, UnitGaussianNormalizer
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    try:
        # 直接用 h5py 按需读取，避免一次性加载整个数据集
        with h5py.File(args.data_path, 'r') as f:
            ds = f['u']  # 原始形状 (S, S, T_total, N)
            S1, S2, T_total, N_total = ds.shape

            ntrain_actual = min(N_total, args.ntrain)
            ntest_actual = min(max(0, N_total - ntrain_actual), args.ntest)

            # 读取训练与测试子集，并排列为 (N, S, S, T)
            train_series = torch.from_numpy(ds[:, :, :, :ntrain_actual]).float().permute(3, 0, 1, 2)
            test_series = torch.from_numpy(ds[:, :, :, N_total - ntest_actual:N_total]).float().permute(3, 0, 1, 2)

        # Sliding windows to amplify samples under limited dataset
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

        # 顺序块训练：步长取 T_out，避免随机滑窗分布漂移
        train_a, train_u, train_starts, train_T = build_windows(train_series, args.T_in, args.T_out, args.T_out, args.max_windows_per_sample)
        test_a, test_u, test_starts, test_T = build_windows(test_series, args.T_in, args.T_out, args.T_out, args.max_windows_per_sample_eval)

        print(f"Data shapes after windows: train_a: {train_a.shape}, train_u: {train_u.shape}")

    except Exception as e:
        print(f"Error loading data from {args.data_path}: {e}")
        return

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    # 绝对时间通道，保留窗口在全局序列的位置
    def append_abs_time(x, starts, T_total, T_in):
        num = x.shape[0]
        S1, S2 = x.shape[1], x.shape[2]
        # 展开为长度为 1 的新通道，拼接后得到 T_in + 1
        t_abs = (starts / float(T_total)).view(num, 1, 1, 1).expand(num, S1, S2, 1)
        return torch.cat([x, t_abs], dim=-1)

    train_a = append_abs_time(train_a, train_starts, train_T, args.T_in)
    test_a = append_abs_time(test_a, test_starts, test_T, args.T_in)

    S1, S2 = train_a.shape[1], train_a.shape[2]
    train_a = train_a.reshape(train_a.shape[0], S1, S2, 1, args.T_in + 1).repeat([1,1,1,args.T_out,1])
    test_a = test_a.reshape(test_a.shape[0], S1, S2, 1, args.T_in + 1).repeat([1,1,1,args.T_out,1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=args.batch_size, shuffle=False)
    # 评估统一在归一化空间：对测试标签做相同的归一化（保持 normalizer 在 CPU）
    test_u = y_normalizer.encode(test_u)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=args.batch_size, shuffle=False)
    # 之后将 normalizer 移到 device，用于解码 raw 指标与后续计算
    y_normalizer.to(device)

    model = FNO_RC_3D(args.modes, args.modes, args.modes, args.width, 
                      in_channels=args.T_in + 1, out_channels=1,
                      num_correction_layers=args.num_correction_layers,
                      cft_L=args.cft_L, cft_M=args.cft_M,
                      correction_scale_init=args.correction_scale_init).to(device)
    
    print(f"\nModel: FNO_RC_3D")
    print(f"Parameters: {count_params(model)}")
    print(f"Hyperparameters: Modes={args.modes}, Width={args.width}, LR={args.learning_rate}, WeightDecay={args.weight_decay}")
    
    # 参数组：主干与 RC 分支使用不同的正则强度
    rc_params, main_params = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ["correction_generator_time", "correction_scale"]):
            rc_params.append(p)
        else:
            main_params.append(p)
    optimizer = Adam([
        {"params": main_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": rc_params, "lr": args.rc_lr, "weight_decay": args.rc_weight_decay}
    ])
    # 与基线一致：使用余弦退火调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_func = LpLoss(size_average=False)
    
    # --- high-frequency regularization helpers ---
    def _build_hf_mask(h: int, w: int, ratio: float, device: torch.device) -> torch.Tensor:
        cy, cx = h // 2, w // 2
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        rr = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
        rmax = rr.max().clamp(min=1.0)
        k_cut = float(ratio) * rmax
        return (rr > k_cut)
    
    print("\n--- Starting Training ---")
    best_test = float('inf')
    patience_left = args.early_stop_patience
    # --- helpers for multi-resolution augmentation ---
    def _spectral_resize_2d(x_hw: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        # x_hw: (B, H, W) real tensor; resize in frequency domain to (target_h, target_w)
        B, H, W = x_hw.shape
        F = torch.fft.fftshift(torch.fft.fft2(x_hw), dim=(-2, -1))
        # resize rows (H -> target_h)
        if target_h == H:
            F_h = F
        elif target_h > H:
            pad_h = target_h - H
            ph_b = pad_h // 2
            ph_a = pad_h - ph_b
            F_h = torch.nn.functional.pad(F, (0, 0, ph_b, ph_a))
        else:
            crop_h = H - target_h
            ch_b = crop_h // 2
            ch_a = crop_h - ch_b
            F_h = F[:, ch_b:H-ch_a, :]
        # resize cols (W -> target_w)
        H2 = F_h.shape[-2]
        if target_w == W:
            F_hw = F_h
        elif target_w > W:
            pad_w = target_w - W
            pw_b = pad_w // 2
            pw_a = pad_w - pw_b
            F_hw = torch.nn.functional.pad(F_h, (pw_b, pw_a, 0, 0))
        else:
            crop_w = W - target_w
            cw_b = crop_w // 2
            cw_a = crop_w - cw_b
            F_hw = F_h[:, :, cw_b:W-cw_a]
        x_res = torch.fft.ifft2(torch.fft.ifftshift(F_hw, dim=(-2, -1))).real
        # scale approximately preserve energy under area change
        x_res = x_res * (target_h / H) * (target_w / W)
        return x_res

    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Multi-resolution augmentation (train only)
            if args.multires_aug:
                target_res = int(np.random.choice(args.aug_resolutions))
                if target_res != x.shape[1] or target_res != x.shape[2]:
                    if args.aug_resample_mode == 'bilinear':
                        # x: (B, S1, S2, T_out, Cin)
                        B, S1, S2, T_out, Cin = x.shape
                        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B*T_out*Cin, 1, S1, S2)
                        x_rs = F.interpolate(x_flat, size=(target_res, target_res), mode='bilinear', align_corners=False)
                        x = x_rs.reshape(B, T_out, Cin, target_res, target_res).permute(0, 3, 4, 1, 2)
                        # y: (B, S1, S2, T_out)
                        y_flat = y.permute(0, 3, 1, 2).reshape(B*T_out, 1, S1, S2)
                        y_rs = F.interpolate(y_flat, size=(target_res, target_res), mode='bilinear', align_corners=False)
                        y = y_rs.reshape(B, T_out, target_res, target_res).permute(0, 2, 3, 1)
                    else:
                        # spectral resize
                        B, S1, S2, T_out, Cin = x.shape
                        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B*T_out*Cin, S1, S2)
                        x_rs = _spectral_resize_2d(x_flat, target_res, target_res)
                        x = x_rs.reshape(B, T_out, Cin, target_res, target_res).permute(0, 3, 4, 1, 2)
                        y_flat = y.permute(0, 3, 1, 2).reshape(B*T_out, S1, S2)
                        y_rs = _spectral_resize_2d(y_flat, target_res, target_res)
                        y = y_rs.reshape(B, T_out, target_res, target_res).permute(0, 2, 3, 1)

            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            # γ warmup：前 args.gamma_warmup_epochs 线性升到 correction_scale_init
            # 冻结 RC 前 warmup_freeze_epochs 个 epoch，之后在 gamma_warmup_epochs 内线性升至目标值
            if ep < args.warmup_freeze_epochs:
                gamma = 0.0
            else:
                warmup_len = max(1, args.gamma_warmup_epochs)
                prog = min(1.0, (ep - args.warmup_freeze_epochs + 1) / warmup_len)
                gamma = prog * args.correction_scale_init
            with torch.no_grad():
                for conv in [model.conv0, model.conv1, model.conv2, model.conv3]:
                    if hasattr(conv, 'correction_scale'):
                        conv.correction_scale.data.fill_(gamma)

            # 公平对比：在归一化空间计算损失（与基线一致）
            loss = loss_func(out, y)
            
            # 高频能量正则：约束 (out - y) 在高频段能量，抑制过冲
            if args.hf_reg_weight > 0.0:
                B, S1b, S2b, Db = out.shape
                diff = (out - y).reshape(B * Db, S1b, S2b)
                Fdiff = torch.fft.fftshift(torch.fft.fft2(diff), dim=(-2, -1))
                P = (Fdiff.real ** 2 + Fdiff.imag ** 2)
                hf_mask = _build_hf_mask(S1b, S2b, args.hf_reg_kcut_ratio, P.device)
                hf_energy = P[:, hf_mask].mean()
                loss = loss + args.hf_reg_weight * hf_energy
            # 时间平滑正则：对校正序列在时间维做一阶差分平滑
            smooth_reg = 0.0
            for conv in [model.conv0, model.conv1]:
                if getattr(conv, 'enable_correction', False) and getattr(conv, 'last_correction', None) is not None:
                    c = conv.last_correction  # (B,out,1,1,D)
                    diff = c[..., 1:] - c[..., :-1]
                    smooth_reg = smooth_reg + torch.mean(diff ** 2)
            loss = loss + args.rc_time_smooth_weight * smooth_reg
            loss.backward()
            
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        raw_test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze(-1)
                # 公平对比：在归一化空间评估
                test_l2 += loss_func(out, y).item()
                # 额外：原始物理尺度指标（不用于优化，仅报告）
                out_raw = y_normalizer.decode(out)
                y_raw = y_normalizer.decode(y)
                raw_test_l2 += loss_func(out_raw, y_raw).item()
        
        train_l2 /= train_a.shape[0]
        test_l2 /= test_a.shape[0]
        raw_test_l2 /= test_a.shape[0]
        
        t2 = default_timer()
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{args.epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f} | Raw Test L2: {raw_test_l2:.6f}")

        # 早停与最佳模型保存（按归一化 Test L2）
        if test_l2 < best_test - 1e-8:
            best_test = test_l2
            patience_left = args.early_stop_patience
            if args.model_save_path:
                best_path = args.model_save_path.replace('.pt', '_best.pt')
                best_dir = os.path.dirname(best_path)
                if best_dir and not os.path.exists(best_dir):
                    os.makedirs(best_dir, exist_ok=True)
                torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {ep+1}. Best Test L2: {best_test:.6f}")
                break
    
    print("--- Training Finished ---")

    if args.model_save_path:
        model_dir = os.path.dirname(args.model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        torch.save(model.state_dict(), args.model_save_path)
        # Save the target normalizer for consistent visualization
        yn_stats = {
            'mean': y_normalizer.mean.detach().cpu(),
            'std':  y_normalizer.std.detach().cpu()
        }
        yn_path = os.path.join(model_dir, 'rc_y_normalizer.pt')
        torch.save(yn_stats, yn_path)
        print(f"Model saved to {args.model_save_path}\nSaved RC y_normalizer stats to {yn_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D FNO-RC.')
    
    parser.add_argument('--data_path', type=str, default='/content/data/ns_V1e-4_N10000_T30.mat')
    parser.add_argument('--model_save_path', type=str, default='models/fno_rc_3d_seq.pt')
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--ntest', type=int, default=200)
    
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_out', type=int, default=20)
    
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_step', type=int, default=100)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--stride', type=int, default=800)
    parser.add_argument('--stride_eval', type=int, default=1000)
    parser.add_argument('--max_windows_per_sample', type=int, default=20)
    parser.add_argument('--max_windows_per_sample_eval', type=int, default=5)
    parser.add_argument('--num_correction_layers', type=int, default=1)
    parser.add_argument('--cft_L', type=int, default=4)
    parser.add_argument('--cft_M', type=int, default=4)
    parser.add_argument('--correction_scale_init', type=float, default=0.02)
    parser.add_argument('--rc_weight_decay', type=float, default=1e-3)
    parser.add_argument('--rc_lr', type=float, default=3e-4)
    parser.add_argument('--rc_time_smooth_weight', type=float, default=3e-3)
    parser.add_argument('--gamma_warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_freeze_epochs', type=int, default=20)
    parser.add_argument('--early_stop_patience', type=int, default=10)

    # Multi-resolution augmentation
    parser.add_argument('--multires_aug', action='store_true')
    parser.add_argument('--aug_resolutions', type=str, default='48,64,80,96')
    parser.add_argument('--aug_resample_mode', type=str, default='spectral', choices=['spectral', 'bilinear'])
    
    # High-frequency regularization
    parser.add_argument('--hf_reg_weight', type=float, default=1e-3)
    parser.add_argument('--hf_reg_kcut_ratio', type=float, default=0.66)
    
    args = parser.parse_args()
    if isinstance(args.aug_resolutions, str):
        args.aug_resolutions = [int(s) for s in args.aug_resolutions.split(',') if s.strip()]
    main(args) 