import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralMixBlock3D(nn.Module):
    """
    Simplified AFNO-style spectral mixing block for 3D tensors.
    Steps: rFFTN → keep low modes → linear mixing on complex spectrum (real/imag split) → irFFTN.
    """
    def __init__(self, channels, modes1, modes2, modes3, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.m1, self.m2, self.m3 = modes1, modes2, modes3
        hidden = int(channels * mlp_ratio)
        # Linear on real/imag separately to avoid complex weights
        self.lin1 = nn.Linear(channels*2, hidden)
        self.lin2 = nn.Linear(hidden, channels*2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B,C,H,W,D)
        B, C, H, W, D = x.shape
        X = torch.fft.rfftn(x, dim=[-3,-2,-1])  # (B,C,H,W,D//2+1)
        Xc = X[:, :, :self.m1, :self.m2, :self.m3]  # low modes (complex)
        # real/imag concat as features
        R = torch.view_as_real(Xc)  # (B,C,m1,m2,m3,2)
        R = R.permute(0,2,3,4,1,5).contiguous().view(B*self.m1*self.m2*self.m3, C*2)
        Hf = self.lin2(F.gelu(self.lin1(R)))
        Hf = self.drop(Hf)
        Hf = Hf.view(B, self.m1, self.m2, self.m3, C, 2).permute(0,4,1,2,3,5).contiguous()
        X_new = torch.zeros_like(X)
        X_new[:, :, :self.m1, :self.m2, :self.m3] = torch.view_as_complex(Hf)
        y = torch.fft.irfftn(X_new, s=(H,W,D))
        return y


class AFNO3D(nn.Module):
    """A light 3D AFNO backbone with residual local paths.
    Input: (B,H,W,D,in_channels) with absolute time channel included.
    Output: (B,H,W,D,1) in raw space.
    """
    def __init__(self, modes1, modes2, modes3, width=32, in_channels=4, out_channels=1, depth=4):
        super().__init__()
        self.width = width
        self.padding = 6
        # Expect in_channels = T_in+1 (absolute time), and we append (x,y,t) internally → +3
        self.fc0 = nn.Linear(in_channels + 3, width)
        blocks = []
        for _ in range(depth):
            blocks.append(nn.ModuleDict({
                'spec': SpectralMixBlock3D(width, modes1, modes2, modes3, mlp_ratio=2.0, drop=0.0),
                'conv': nn.Conv3d(width, width, 1)
            }))
        self.blocks = nn.ModuleList(blocks)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B,H,W,D,in_channels) where channels = T_in+1 (abs time)
        B, H, W, D, _ = x.shape
        # Append coordinates (x,y,t_rel)
        gx = torch.linspace(0, 1, H, device=x.device).view(1, H, 1, 1, 1).repeat(B, 1, W, D, 1)
        gy = torch.linspace(0, 1, W, device=x.device).view(1, 1, W, 1, 1).repeat(B, H, 1, D, 1)
        gt = torch.linspace(0, 1, D, device=x.device).view(1, 1, 1, D, 1).repeat(B, H, W, 1, 1)
        x = torch.cat([x, gx, gy, gt], dim=-1)
        x = self.fc0(x)
        x = x.permute(0,4,1,2,3)  # (B,C,H,W,D)
        x = F.pad(x, [0,self.padding,0,self.padding,0,self.padding])
        for b in self.blocks:
            xs = b['spec'](x)
            xr = b['conv'](x)
            x = F.gelu(xs + xr)
        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0,2,3,4,1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


