"""
This script is a standalone version of the 1D Dynamic Filter FNO experiment,
designed to be run directly in a Google Colab notebook.

It includes all necessary dependencies (`Adam`, `utilities3`) and the
core model logic in a single file.

To run this script in Colab:
1.  Create a new Colab notebook.
2.  Copy and paste the entire content of this file into a single code cell.
3.  Upload the 'burgers_data_R10.mat' file to the Colab session storage.
4.  Run the cell.
"""

# ==============================================================================
# 1. Dependencies and Imports
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import numpy as np
import scipy.io
import h5py
import mat73
import matplotlib.pyplot as plt
import os
import math
from timeit import default_timer
from typing import List, Optional
from torch.optim.optimizer import Optimizer


# ==============================================================================
# 2. Dependency: Adam Optimizer (from Adam.py)
# ==============================================================================
def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation."""
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)

class Adam(Optimizer):
    """Implements Adam algorithm."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    state['step'] += 1
                    state_steps.append(state['step'])
            adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                 amsgrad=group['amsgrad'], beta1=beta1, beta2=beta2, lr=group['lr'],
                 weight_decay=group['weight_decay'], eps=group['eps'])
        return loss

# ==============================================================================
# 4. Dependency: Utilities (from utilities3.py)
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self._load_file()

    def _load_file(self):
        # Based on our check, 'burgers.mat' is a v7.3 file.
        # We will use h5py directly, which is the correct reader.
        try:
            self.data = h5py.File(self.file_path, 'r')
        except Exception as e:
            print(f"Failed to load MAT v7.3 file using h5py. The specific error is:")
            raise e

    def read_field(self, field):
        x = self.data[field]
        x = x[()] # For h5py, we need to extract the data
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.to(device)
        return x

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

# ==============================================================================
# 5. Core Logic: Vectorized CFT and FNO Model
# ==============================================================================
torch.manual_seed(0)
np.random.seed(0)

def vectorized_batched_cft_decomposed(x, t, freqs, L, M):
    if x.is_complex():
        raise NotImplementedError("Real-decomposed CFT currently only supports real input signals.")
    batch_size, in_channels, n_samples = x.shape
    n_freqs = len(freqs)
    device = x.device
    total_spectrum_real = torch.zeros(batch_size, in_channels, n_freqs, device=device, dtype=torch.float32)
    total_spectrum_imag = torch.zeros(batch_size, in_channels, n_freqs, device=device, dtype=torch.float32)
    segment_len = 1.0 / L
    segment_starts = torch.linspace(0, 1, L + 1, device=device)[:-1]
    k = torch.arange(M, device=device)
    cheb_nodes_ref = -torch.cos((2 * k + 1) * torch.pi / (2 * M))
    T_k_at_nodes = torch.cos(k.unsqueeze(1) * torch.acos(cheb_nodes_ref.unsqueeze(0)))
    freq_factor = 2 * torch.pi * (segment_len / 2)
    w_prime = freqs.unsqueeze(0) * freq_factor
    cheb_nodes_grid, w_prime_grid = torch.meshgrid(cheb_nodes_ref, w_prime.squeeze(0), indexing='ij')
    angle_quad = cheb_nodes_grid * w_prime_grid
    exp_term_real = torch.cos(angle_quad)
    exp_term_imag = -torch.sin(angle_quad)
    cheb_weights_real = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_real) * (segment_len / 2)
    cheb_weights_imag = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_imag) * (segment_len / 2)
    for i in range(L):
        a = segment_starts[i]
        t_segment = a + (segment_len / 2) * (cheb_nodes_ref + 1)
        right_indices = torch.searchsorted(t, t_segment).clamp(max=n_samples - 1)
        left_indices = (right_indices - 1).clamp(min=0)
        t_left = t[left_indices]
        t_right = t[right_indices]
        denom = t_right - t_left
        denom[denom == 0] = 1.0
        w_right = (t_segment - t_left) / denom
        w_left = 1.0 - w_right
        signal_left = x[:, :, left_indices]
        signal_right = x[:, :, right_indices]
        signal_segments = w_left * signal_left + w_right * signal_right
        spectrum_segment_real = torch.einsum("bcm,mf->bcf", signal_segments, cheb_weights_real)
        spectrum_segment_imag = torch.einsum("bcm,mf->bcf", signal_segments, cheb_weights_imag)
        angle_shift = 2 * torch.pi * freqs * a
        exp_shift_real = torch.cos(angle_shift)
        exp_shift_imag = -torch.sin(angle_shift)
        total_spectrum_real += (spectrum_segment_real * exp_shift_real) - (spectrum_segment_imag * exp_shift_imag)
        total_spectrum_imag += (spectrum_segment_real * exp_shift_imag) + (spectrum_segment_imag * exp_shift_real)
    return torch.complex(total_spectrum_real, total_spectrum_imag)

class SpectralConv1d_Interpolate(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d_Interpolate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights1_A = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights1_B = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d_dynamic(self, input, weights):
        return torch.einsum("bix,bioy->box", input, weights)

    def forward(self, x, gate):
        batchsize = x.shape[0]
        gate_reshaped = gate.view(batchsize, 1, self.out_channels, 1)
        weights1 = gate_reshaped * self.weights1_A + (1 - gate_reshaped) * self.weights1_B
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d_dynamic(x_ft[:, :, :self.modes1], weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class DynamicFNO1d(nn.Module):
    def __init__(self, modes, width):
        super(DynamicFNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)
        self.gate_net0 = nn.Linear(self.modes1, self.width)
        self.gate_net1 = nn.Linear(self.modes1, self.width)
        self.gate_net2 = nn.Linear(self.modes1, self.width)
        self.gate_net3 = nn.Linear(self.modes1, self.width)
        self.conv0 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d_Interpolate(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_gate_signal(self, x, gate_net):
        n_samples = x.shape[2]
        t_coords = torch.linspace(0, 1, n_samples, device=x.device)
        f_points = torch.fft.rfftfreq(n_samples, d=1/n_samples).to(x.device)
        x_cft = vectorized_batched_cft_decomposed(x=x, t=t_coords, freqs=f_points, L=20, M=8).detach()
        cft_mag = torch.abs(x_cft[:, :, :self.modes1])
        cft_mag_avg = torch.mean(cft_mag, dim=1)
        gate = torch.sigmoid(gate_net(cft_mag_avg))
        return gate

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        gate0 = self.get_gate_signal(x, self.gate_net0)
        x1 = self.conv0(x, gate0)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        gate1 = self.get_gate_signal(x, self.gate_net1)
        x1 = self.conv1(x, gate1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        gate2 = self.get_gate_signal(x, self.gate_net2)
        x1 = self.conv2(x, gate2)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        gate3 = self.get_gate_signal(x, self.gate_net3)
        x1 = self.conv3(x, gate3)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

# ==============================================================================
# 6. Main Execution Block
# ==============================================================================
# --- Configurations ---
ntrain = 1000
ntest = 100
sub = 2**3
h = 2**13 // sub
s = h
batch_size = 20
learning_rate = 0.001
epochs = 50
step_size = 100
gamma = 0.5
modes = 16
width = 64
DATA_PATH = 'burgers2.mat'

# --- Data Loading ---
print("Loading data...")
try:
    dataloader = MatReader(DATA_PATH)
    x_data = dataloader.read_field('input')
    # The 'output' field contains the time series u(t,x)
    # We need to transpose it to (time, space)
    y_data_series = dataloader.read_field('output').T
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: Data file not found at '{DATA_PATH}'")
except Exception as e:
    print(f"An error occurred while loading data: {e}")

if 'y_data_series' in locals():
    # The dataset contains one long time series. We'll create sample pairs
    # where an input x = u(t) maps to an output y = u(t+1).
    x_data = y_data_series[0, :-1, :]  # All timesteps except the last
    y_data = y_data_series[0, 1:, :]   # All timesteps except the first

    # Use all available pairs for training and a small validation set
    ntrain = x_data.shape[0]
    s = x_data.shape[1]
    
    print(f"Reconstructed data with {ntrain} samples from time series.")
    print(f"Spatial resolution (s): {s}")

    x_train = x_data.reshape(ntrain, s, 1)
    y_train = y_data
    
    ntest = 200
    x_test = x_train[-ntest:]
    y_test = y_train[-ntest:]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # --- Model Initialization ---
    print("Initializing model...")
    model = DynamicFNO1d(modes, width).to(device)
    print(f"Using device: {device}")
    # print(f"Model parameters: {count_params(model)}")

    # --- Training ---
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = LpLoss(size_average=False)
    best_test_l2 = float('inf')

    print(f"Starting training for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
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
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest
        
        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            # Optional: Save best model state
            # torch.save(model.state_dict(), 'best_model_1d_dynamic.pth')

        t2 = default_timer()
        print(f"Epoch: {ep}, Time: {t2-t1:.4f}, Train L2: {train_l2:.6f}, Test L2: {test_l2:.6f}")

    print("Training finished.")
    print(f"\nBest Test L2 Loss: {best_test_l2:.6f}") 