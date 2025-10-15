import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d_U(nn.Module):
    """A standard spectral convolution layer used in the U-FNO architecture."""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_U, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        # x shape: (batch, in_channels, x, y, z)
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights)

        x = torch.fft.irfftn(out_ft, s=[x.size(-3), x.size(-2), x.size(-1)])
        return x

class U_FNO_3d(nn.Module):
    """
    A 3D U-Net-style Fourier Neural Operator.
    It processes data at multiple resolutions (scales) using an
    encoder-decoder structure with skip connections.
    """
    def __init__(self, modes1, modes2, modes3, width, in_channels=10, out_channels=10):
        super(U_FNO_3d, self).__init__()
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.width = width
        self.in_channels = in_channels  # Input time steps
        self.out_channels = out_channels # Output time steps

        # Lifting layer
        self.fc0 = nn.Linear(self.in_channels, self.width)

        # Encoder path
        self.enc0_conv = self._make_layer(self.width, self.width)
        self.enc1_conv = self._make_layer(self.width, self.width)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = self._make_layer(self.width, self.width)

        # Decoder path
        self.dec1_conv = self._make_layer(self.width * 2, self.width)
        self.dec0_conv = self._make_layer(self.width * 2, self.width)

        # Projection to output
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def _make_layer(self, in_c, out_c):
        """Helper to create a standard FNO block."""
        conv = SpectralConv3d_U(in_c, out_c, self.modes1, self.modes2, self.modes3)
        w = nn.Conv3d(in_c, out_c, 1)
        return nn.ModuleDict({'conv': conv, 'w': w})

    def _apply_layer(self, x, layer):
        """Helper to apply an FNO block."""
        x_spec = layer['conv'](x)
        x_skip = layer['w'](x)
        x = x_spec + x_skip
        return F.gelu(x)

    def forward(self, x):
        # x: (batch, H, W, D, T_in)
        
        # --- Lifting ---
        x = self.fc0(x) # -> (b, H, W, D, width)
        x = x.permute(0, 4, 1, 2, 3) # -> (b, width, H, W, D)

        # --- Encoder ---
        e0 = self._apply_layer(x, self.enc0_conv)
        e1 = self._apply_layer(e0, self.enc1_conv)
        
        # --- Bottleneck ---
        b_in = self.pool(e1)
        b_out = self._apply_layer(b_in, self.bottleneck_conv)

        # --- Decoder ---
        d1_in = F.interpolate(b_out, size=e1.shape[-3:], mode='trilinear', align_corners=False)
        d1_in = torch.cat([d1_in, e1], dim=1)
        d1_out = self._apply_layer(d1_in, self.dec1_conv)
        
        d0_in = F.interpolate(d1_out, size=e0.shape[-3:], mode='trilinear', align_corners=False)
        d0_in = torch.cat([d0_in, e0], dim=1)
        d0_out = self._apply_layer(d0_in, self.dec0_conv)

        # --- Projection ---
        x = d0_out.permute(0, 2, 3, 4, 1) # -> (b, H, W, D, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # -> (b, H, W, D, T_out)
        
        return x 