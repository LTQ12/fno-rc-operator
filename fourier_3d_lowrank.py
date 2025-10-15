import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv3dLowRank(nn.Module):
    """
    3D Spectral Convolution with low-rank factorization on channel mixing.
    weights ≈ U (in_channels,r) · B (r,m1,m2,m3) · V^T (r,out_channels)
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, rank=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.rank = rank

        scale = 1.0 / np.sqrt(in_channels * out_channels)
        # Channel factors (real)
        self.U = nn.Parameter(scale * torch.randn(in_channels, rank))
        self.V = nn.Parameter(scale * torch.randn(out_channels, rank))
        # Frequency core (complex)
        self.B = nn.Parameter(scale * torch.randn(rank, modes1, modes2, modes3, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, C_in, H, W, D)
        B = x.shape[0]
        H, W, D = x.size(-3), x.size(-2), x.size(-1)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])  # (B,C_in,H,W,D//2+1)

        # Restrict to modes
        x_ft_crop = x_ft[:, :, :self.modes1, :self.modes2, :self.modes3]  # (B,C_in,m1,m2,m3)

        # 保持复数计算：将通道因子提升为复数
        U_c = self.U.to(dtype=torch.cfloat)
        V_c = self.V.to(dtype=torch.cfloat)
        # (B,C_in,m1,m2,m3) × (C_in, r) -> (B,r,m1,m2,m3)
        xr = torch.einsum('bcxyz,cr->brxyz', x_ft_crop, U_c)
        # Multiply frequency core (r,m1,m2,m3) (complex)
        y_r = xr * self.B  # broadcasting on r,m1,m2,m3
        # (B,r,m1,m2,m3) × (C_out,r) -> (B,C_out,m1,m2,m3)
        y = torch.einsum('brxyz,or->boxyz', y_r, V_c)

        # Pad back to full spectrum grid
        out_ft = torch.zeros(B, self.out_channels, H, W, D//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = y

        out = torch.fft.irfftn(out_ft, s=(H, W, D))
        return out


class LowRankFNO3d(nn.Module):
    """
    Low-Rank FNO-3D baseline (raw-space output). Input包含绝对时间通道+坐标（由上层拼接）。
    """
    def __init__(self, modes1, modes2, modes3, width=32, in_channels=4, out_channels=1, rank=8):
        super().__init__()
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.width = width
        self.padding = 6

        self.fc0 = nn.Linear(in_channels, width)

        self.conv0 = SpectralConv3dLowRank(width, width, modes1, modes2, modes3, rank)
        self.conv1 = SpectralConv3dLowRank(width, width, modes1, modes2, modes3, rank)
        self.conv2 = SpectralConv3dLowRank(width, width, modes1, modes2, modes3, rank)
        self.conv3 = SpectralConv3dLowRank(width, width, modes1, modes2, modes3, rank)
        self.w0 = nn.Conv3d(width, width, 1)
        self.w1 = nn.Conv3d(width, width, 1)
        self.w2 = nn.Conv3d(width, width, 1)
        self.w3 = nn.Conv3d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, H, W, D, in_channels)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B,C,H,W,D)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        x1 = self.conv0(x); x2 = self.w0(x); x = F.gelu(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = F.gelu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = F.gelu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = x1 + x2

        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # (B,H,W,D,C)
        x = self.fc1(x); x = F.gelu(x); x = self.fc2(x)
        return x


