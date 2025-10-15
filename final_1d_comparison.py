"""
================================================================================
FINAL 1D COMPARISON: FNO vs. G-FNO vs. Dynamic-FNO
================================================================================

This script serves as the final conclusive experiment for the 1D part of our project.
It directly compares three key architectures on the same time-series dataset (`burgers2.mat`):

1.  **FNO1d_FFT**: A standard, baseline Fourier Neural Operator using pure FFT.
2.  **GatedFNO1d**: Our first enhancement, which uses a CFT-generated gate to modulate
    the *frequency signal* before the spectral convolution.
3.  **DynamicFNO1d**: Our ultimate architecture, which uses a CFT-generated gate to
    dynamically *interpolate the filter weights* themselves for each sample.

The script is fully standalone. It will train and evaluate each model sequentially
and print a final report comparing their best-achieved test errors.
"""

# ==============================================================================
# 1. Standard Imports and Dependencies
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
import matplotlib.pyplot as plt
import os
import math
from timeit import default_timer
from typing import List, Optional
from torch.optim.optimizer import Optimizer
from functools import reduce
import operator

# Set a seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ==============================================================================
# 2. Dependency: Adam Optimizer
# (Standard implementation, included for self-containment)
# ==============================================================================
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = [], [], [], [], [], []
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse: raise RuntimeError('Adam does not support sparse gradients')
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
                    if group['amsgrad']: max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    state['step'] += 1
                    state_steps.append(state['step'])
            # Functional API call
            F_adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                   amsgrad=group['amsgrad'], beta1=beta1, beta2=beta2, lr=group['lr'],
                   weight_decay=group['weight_decay'], eps=group['eps'])
        return loss

def F_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor],
           max_exp_avg_sqs: List[Tensor], state_steps: List[int], *, amsgrad: bool, beta1: float, beta2: float,
           lr: float, weight_decay: float, eps: float):
    for i, param in enumerate(params):
        grad, exp_avg, exp_avg_sq, step = grads[i], exp_avgs[i], exp_avg_sqs[i], state_steps[i]
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        if weight_decay != 0: grad = grad.add(param, alpha=weight_decay)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)

# ==============================================================================
# 3. Dependency: Utilities
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.file_path = file_path
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.data = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = h5py.File(self.file_path, 'r')
        except Exception as e:
            print(f"Failed to load MAT v7.3 file using h5py. Trying scipy.io fallback.")
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except Exception as e_scipy:
                raise IOError(f"Could not read file {self.file_path} with either h5py or scipy. Errors: \nh5py: {e}\nscipy: {e_scipy}")

    def read_field(self, field):
        x = self.data[field]
        if isinstance(x, np.ndarray): # Scipy loads as numpy array
            pass
        else: # h5py loads as h5py object
            x = x[()]
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.to(device)
        return x

class LpLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average
    def rel(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        if self.size_average:
            return torch.mean(diff_norms/y_norms)
        else:
            return torch.sum(diff_norms/y_norms)
    def __call__(self, x, y):
        return self.rel(x, y)

# ==============================================================================
# 4. Dependency: Vectorized CFT
# ==============================================================================
def vectorized_batched_cft_decomposed(x, t, freqs, L, M):
    if x.is_complex(): raise NotImplementedError("Real-decomposed CFT currently only supports real input signals.")
    batch_size, in_channels, n_samples = x.shape
    n_freqs, device = len(freqs), x.device
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
    exp_term_real, exp_term_imag = torch.cos(angle_quad), -torch.sin(angle_quad)
    cheb_weights_real = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_real) * (segment_len / 2)
    cheb_weights_imag = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_imag) * (segment_len / 2)
    for i in range(L):
        a = segment_starts[i]
        t_segment = a + (segment_len / 2) * (cheb_nodes_ref + 1)
        right_indices = torch.searchsorted(t, t_segment).clamp(max=n_samples - 1)
        left_indices = (right_indices - 1).clamp(min=0)
        t_left, t_right = t[left_indices], t[right_indices]
        denom = t_right - t_left
        denom[denom == 0] = 1.0
        w_right = (t_segment - t_left) / denom
        w_left = 1.0 - w_right
        signal_left, signal_right = x[:, :, left_indices], x[:, :, right_indices]
        signal_segments = w_left * signal_left + w_right * signal_right
        spectrum_segment_real = torch.einsum("bcm,mf->bcf", signal_segments, cheb_weights_real)
        spectrum_segment_imag = torch.einsum("bcm,mf->bcf", signal_segments, cheb_weights_imag)
        angle_shift = 2 * torch.pi * freqs * a
        exp_shift_real, exp_shift_imag = torch.cos(angle_shift), -torch.sin(angle_shift)
        total_spectrum_real += (spectrum_segment_real * exp_shift_real) - (spectrum_segment_imag * exp_shift_imag)
        total_spectrum_imag += (spectrum_segment_real * exp_shift_imag) + (spectrum_segment_imag * exp_shift_real)
    return torch.complex(total_spectrum_real, total_spectrum_imag)

# ==============================================================================
# 5. MODEL ARCHITECTURES
# ==============================================================================

# ------------------------------------------------------------------------------
# ARCHITECTURE 1: Standard FNO (FFT-based)
# ------------------------------------------------------------------------------
class SpectralConv1d_FFT(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d_FFT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d_FFT(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d_FFT, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)
        self.conv0, self.conv1, self.conv2, self.conv3 = [SpectralConv1d_FFT(self.width, self.width, self.modes1) for _ in range(4)]
        self.w0, self.w1, self.w2, self.w3 = [nn.Conv1d(self.width, self.width, 1) for _ in range(4)]
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx.to(device)

# ------------------------------------------------------------------------------
# ARCHITECTURE 2: Gated FNO (Signal Gating)
# ------------------------------------------------------------------------------
class GatedSpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(GatedSpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.gate_transform = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.gate_norm = nn.LayerNorm(modes1)

    def forward(self, x):
        batchsize, n_samples = x.shape[0], x.shape[2]
        t_coords = torch.linspace(0, 1, n_samples, device=x.device)
        f_points = torch.fft.rfftfreq(n_samples, d=1/n_samples).to(x.device)
        x_fft = torch.fft.rfft(x)
        x_cft = vectorized_batched_cft_decomposed(x, t_coords, f_points, L=20, M=8).detach()
        cft_magnitude = torch.abs(x_cft[:, :, :self.modes1])
        gate_normalized = self.gate_norm(cft_magnitude)
        gate = torch.sigmoid(self.gate_transform(gate_normalized))
        gated_fft_modes = x_fft[:, :, :self.modes1] * gate
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", gated_fft_modes, self.weights1)
        x = torch.fft.irfft(out_ft, n=n_samples)
        return x

class GatedFNO1d(nn.Module):
    def __init__(self, modes, width):
        super(GatedFNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)
        self.conv0, self.conv1, self.conv2, self.conv3 = [GatedSpectralConv1d(self.width, self.width, self.modes1) for _ in range(4)]
        self.w0, self.w1, self.w2, self.w3 = [nn.Conv1d(self.width, self.width, 1) for _ in range(4)]
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx.to(device)

# ------------------------------------------------------------------------------
# ARCHITECTURE 3: Dynamic Filter FNO (Filter Gating/Interpolation)
# ------------------------------------------------------------------------------
class SpectralConv1d_Interpolate(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d_Interpolate, self).__init__()
        self.in_channels, self.out_channels, self.modes1 = in_channels, out_channels, modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights1_A = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights1_B = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
    def forward(self, x, gate):
        batchsize = x.shape[0]
        gate_reshaped = gate.view(batchsize, 1, self.out_channels, 1)
        weights1 = gate_reshaped * self.weights1_A + (1 - gate_reshaped) * self.weights1_B
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,bioy->box", x_ft[:, :, :self.modes1], weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class DynamicFNO1d(nn.Module):
    def __init__(self, modes, width):
        super(DynamicFNO1d, self).__init__()
        self.modes1, self.width = modes, width
        self.fc0 = nn.Linear(2, self.width)
        self.gate_nets = nn.ModuleList([nn.Linear(self.modes1, self.width) for _ in range(4)])
        self.convs = nn.ModuleList([SpectralConv1d_Interpolate(self.width, self.width, self.modes1) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(4)])
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
        x = self.fc0(x).permute(0, 2, 1)
        for i in range(4):
            gate = self.get_gate_signal(x, self.gate_nets[i])
            x1 = self.convs[i](x, gate)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < 3: x = F.gelu(x)
        x = x.permute(0, 2, 1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx.to(device)


# ==============================================================================
# 6. UNIFIED TRAINING AND EVALUATION HARNESS
# ==============================================================================
def run_experiment(model, model_name, train_loader, test_loader, config):
    """
    Trains and evaluates a given model.
    """
    print("-" * 60)
    print(f"STARTING EXPERIMENT FOR: {model_name}")
    print("-" * 60)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    loss_fn = LpLoss(size_average=False)
    
    best_test_l2 = float('inf')
    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    for ep in range(config['epochs']):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out.view(x.size(0), -1), y.view(x.size(0), -1))
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
                test_l2 += loss_fn(out.view(x.size(0), -1), y.view(x.size(0), -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest
        
        if test_l2 < best_test_l2:
            best_test_l2 = test_l2

        if (ep + 1) % 10 == 0:
             print(f"Epoch: {ep+1}, Time: {default_timer()-t1:.2f}s, Train L2: {train_l2:.6f}, Test L2: {test_l2:.6f}")
    
    print(f"TRAINING COMPLETE for {model_name}")
    print(f"Best Test L2 Loss: {best_test_l2:.6f}")
    print("-" * 60 + "\n")
    return best_test_l2

# ==============================================================================
# 7. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- Unified Configurations ---
    CONFIG = {
        'data_path': 'burgers2.mat',
        'modes': 16,
        'width': 64,
        'batch_size': 20,
        'epochs': 50,
        'learning_rate': 0.001,
        'step_size': 100,
        'gamma': 0.5,
    }

    # --- Data Loading ---
    print("Loading and preparing data...")
    try:
        dataloader = MatReader(CONFIG['data_path'])
        # The 'output' field from .mat can have shape (1, space, time), (space, time, 1), etc.
        data = dataloader.read_field('output')
        
        # Squeeze out all singleton dimensions to get a 2D tensor of (space, time)
        data = data.squeeze()

        # After squeezing, we must have a 2D tensor.
        if data.dim() != 2:
            raise ValueError(f"Data in .mat file has unexpected shape after squeeze: {data.shape}. Expected 2D.")

        # Transpose to (time, space)
        data_series = data.T 

        # Create input/output pairs for time-stepping: x=u(t), y=u(t+1)
        x_data = data_series[:-1]  # All timesteps except the last
        y_data = data_series[1:]   # All timesteps except the first

        ntrain_total, s = x_data.shape
        
        # Split data into training and testing sets
        ntest = int(ntrain_total * 0.2) # Use 20% for testing
        ntrain = ntrain_total - ntest

        # Reshape for the model: (samples, space, channels)
        x_full = x_data.reshape(ntrain_total, s, 1)
        y_full = y_data.reshape(ntrain_total, s, 1)

        x_train = x_full[:ntrain]
        y_train = y_full[:ntrain]
        x_test = x_full[-ntest:]
        y_test = y_full[-ntest:]

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
        print(f"Data ready. Using {s} spatial resolution.")
        print(f"Training on {len(x_train)} samples, testing on {len(x_test)} samples.")

    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}")
        exit()

    # --- Model Initialization ---
    fno_model = FNO1d_FFT(modes=CONFIG['modes'], width=CONFIG['width'])
    gated_fno_model = GatedFNO1d(modes=CONFIG['modes'], width=CONFIG['width'])
    dynamic_fno_model = DynamicFNO1d(modes=CONFIG['modes'], width=CONFIG['width'])

    models_to_test = {
        "Standard FNO (FFT)": fno_model,
        "Gated FNO (Signal Gating)": gated_fno_model,
        "Dynamic Filter FNO": dynamic_fno_model,
    }

    results = {}

    # --- Run All Experiments ---
    for name, model in models_to_test.items():
        best_loss = run_experiment(model, name, train_loader, test_loader, CONFIG)
        results[name] = best_loss

    # --- Final Report ---
    print("=" * 60)
    print("           FINAL 1D COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model Architecture':<35} | {'Best Test L2 Loss'}")
    print("-" * 60)
    
    baseline_loss = results.get("Standard FNO (FFT)", float('inf'))

    for name, loss in results.items():
        improvement_str = ""
        if name != "Standard FNO (FFT)" and baseline_loss != float('inf'):
            improvement = ((baseline_loss - loss) / baseline_loss) * 100
            improvement_str = f"({improvement:+.2f}%)"
        print(f"{name:<35} | {loss:.6f} {improvement_str}")
    
    print("=" * 60)
    print("Comparison complete.") 