import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Pruned from https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

def cft1d(x, modes1, L_segments=4, M_cheb=8):
    """
    Computes the 1D Continuous Fourier Transform (CFT) coefficients for a batch of signals.
    """
    batch_size, S, in_channels = x.shape
    x_permuted = x.permute(0, 2, 1) # (batch_size, in_channels, S)
    
    T_list = []
    for l in range(L_segments):
        T_l_list = []
        for m in range(M_cheb):
            cheb_poly = np.polynomial.chebyshev.Chebyshev.basis(m)
            t_vals = np.linspace(-1, 1, S)
            T_lm = torch.tensor(cheb_poly(t_vals), dtype=torch.float32).to(x.device)
            T_l_list.append(T_lm)
        T_list.append(torch.stack(T_l_list, dim=0))
    # Cast the Chebyshev tensor to complex to match the f_exp tensor in einsum
    T = torch.stack(T_list, dim=0).to(torch.cfloat)

    c_coeffs_list = []
    for k in range(-modes1, modes1 + 1):
        exp_term = torch.exp(-1j * np.pi * k * torch.arange(S, device=x.device) / S)
        f_exp = x_permuted * exp_term.view(1, 1, -1)
        
        c_k_list = []
        for l in range(L_segments):
            # Use einsum for a clear, batch-wise dot product over each channel.
            # 'bcs,ms->bcm' means: for each batch 'b' and channel 'c', compute the dot
            # product of the signal 's' with each Chebyshev mode 'm' over the spatial dim 's'.
            integral_approx = torch.einsum('bcs,ms->bcm', f_exp, T[l, :, :]) / S
            c_k_list.append(integral_approx)
        # Resulting shape of stack: (batch, in_channels, M_cheb, L_segments)
        c_coeffs_list.append(torch.stack(c_k_list, dim=-1))
    
    # Shape of c_coeffs: (batch, in_channels, M_cheb, L_segments, 2*modes1+1)
    c_coeffs = torch.stack(c_coeffs_list, dim=-1)
    # Flatten all dimensions except batch for the linear layer
    c_coeffs_real = torch.view_as_real(c_coeffs).flatten(start_dim=1)
    return c_coeffs_real

class FNO_RC_1D(nn.Module):
    def __init__(self, modes, width, in_channels=1, out_channels=1):
        super(FNO_RC_1D, self).__init__()
        self.modes1 = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(self.in_channels + 1, self.width) # input channel is 2: (u(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

        # CFT-based Residual Correction Path
        self.cft_modes1 = 4 
        self.L_segments = 2
        self.M_cheb = 4
        # The CFT is applied on the lifted signal which has `width` channels
        cft_flat_dim = (2 * self.cft_modes1 + 1) * self.L_segments * self.M_cheb * self.width * 2

        self.correction_generator = nn.Sequential(
            nn.Linear(cft_flat_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.width)
        )
        
        # Zero-initialize the final layer of the correction generator
        with torch.no_grad():
            self.correction_generator[-1].weight.zero_()
            self.correction_generator[-1].bias.zero_()

    def get_grid(self, shape, device):
        batchsize, size = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size), dtype=torch.float)
        gridx = gridx.reshape(1, size, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def forward(self, x):
        # x shape: (batch, size, T_in)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # (batch, size, T_in + 1)
        x = self.fc0(x) # (batch, size, width)
        x = x.permute(0, 2, 1) # (batch, width, size)

        # Main FNO Path (Prediction)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        fno_pred = x1 + x2 # This is v_t+1_prime in the diagram

        # CFT Path (Correction)
        cft_coeffs = cft1d(x.permute(0, 2, 1), self.cft_modes1, self.L_segments, self.M_cheb)
        correction_latent = self.correction_generator(cft_coeffs) # (batch, width)
        correction_field = correction_latent.unsqueeze(-1) # (batch, width, 1) to allow broadcasting
        
        # Combine Prediction and Correction
        x = fno_pred + correction_field
        x = x.permute(0, 2, 1) # (batch, size, width)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # (batch, size, T_out)
        return x 