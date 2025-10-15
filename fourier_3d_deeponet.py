import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BranchNet3D(nn.Module):
    """
    A light 3D CNN to encode the input window (S1,S2,T_in) into a global latent code of size p.
    Input:  (B, 1, S1, S2, T_in)
    Output: (B, p)
    """
    def __init__(self, in_channels=1, base=16, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, base, 3, padding=1)
        self.conv2 = nn.Conv3d(base, base, 3, padding=1)
        self.conv3 = nn.Conv3d(base, base*2, 3, padding=1, stride=(2,2,2))
        self.conv4 = nn.Conv3d(base*2, base*2, 3, padding=1)
        self.conv5 = nn.Conv3d(base*2, base*4, 3, padding=1, stride=(2,2,2))
        self.conv6 = nn.Conv3d(base*4, base*4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base*4, latent_dim)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x  # (B, p)


class TrunkNet(nn.Module):
    """
    MLP that maps (x,y,rel_t,t_abs) → R^p for each query point.
    Input:  (N_pts, 4)
    Output: (N_pts, p)
    """
    def __init__(self, latent_dim=128, hidden=256, depth=3):
        super().__init__()
        layers = []
        in_dim = 4
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i==0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)


class DeepONet3D(nn.Module):
    """
    DeepONet for 3D (2D space + time) prediction.
    Branch: 3D CNN on input window to produce a latent vector (B,p).
    Trunk:  MLP on (x,y,rel_t,t_abs) to produce latent features (N_pts,p).
    Output field constructed by dot-product: u(x,y,t) = <branch_code, trunk_feat(x,y,t)>.

    Forward expects:
      - branch_input: (B,1,S1,S2,T_in)
      - grid_xy:      (S1,S2,2) values in [0,1]
      - rel_times:    (T_out,) values in [0,1]
      - t_abs:        (B,) absolute window start fraction in [0,1]
    Returns:
      - (B,S1,S2,T_out)
    """
    def __init__(self, latent_dim=128, trunk_hidden=256, trunk_depth=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.branch = BranchNet3D(in_channels=1, base=16, latent_dim=latent_dim)
        self.trunk = TrunkNet(latent_dim=latent_dim, hidden=trunk_hidden, depth=trunk_depth)

    @staticmethod
    def get_grid_xy(S1, S2, device):
        gx = torch.linspace(0, 1, S1, device=device)
        gy = torch.linspace(0, 1, S2, device=device)
        gridx = gx.view(S1,1).repeat(1,S2)
        gridy = gy.view(1,S2).repeat(S1,1)
        return torch.stack([gridx, gridy], dim=-1)  # (S1,S2,2)

    def forward(self, branch_input, grid_xy, rel_times, t_abs):
        B = branch_input.shape[0]
        S1, S2 = grid_xy.shape[0], grid_xy.shape[1]
        T_out = rel_times.shape[0]

        device = branch_input.device

        # Branch code (B,p)
        code = self.branch(branch_input)  # (B,p)

        # Precompute (x,y,rel_t)
        xy = grid_xy.view(-1, 2)                      # (S1*S2,2)
        rt = rel_times.view(-1,1)                     # (T_out,1)
        xy_rep = xy.unsqueeze(1).repeat(1, T_out, 1).view(-1,2)  # (S1*S2*T_out,2)
        rt_rep = rt.repeat(S1*S2, 1)                               # (S1*S2*T_out,1)

        # For each sample, append its t_abs and run trunk
        outputs = []
        scale = 1.0 / np.sqrt(self.latent_dim)
        for b in range(B):
            tab_b = torch.full((S1*S2*T_out, 1), float(t_abs[b].item()), device=device)
            coords_b = torch.cat([xy_rep.to(device), rt_rep.to(device), tab_b], dim=1)  # (N_pts,4)
            feat_b = self.trunk(coords_b)  # (N_pts,p)
            # Dot with branch code of this sample
            y_b = feat_b @ code[b].unsqueeze(-1) * scale   # (N_pts,1)
            outputs.append(y_b.view(S1, S2, T_out))

        y = torch.stack(outputs, dim=0)  # (B,S1,S2,T_out)
        return y


