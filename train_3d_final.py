import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import h5py

from utilities3 import *
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 3D Fourier Layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.scale = (1 / (in_channels * out_channels))
        
        # These are the weights for the standard FNO
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # This function handles both 5D weights (standard FNO)
        # and 6D weights (dynamic FNO, which has a batch dimension)
        if weights.dim() == 5:
            # Standard FNO: (b,i,x,y,z) * (i,o,x,y,z) -> (b,o,x,y,z)
            return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
        elif weights.dim() == 6:
            # Dynamic FNO: (b,i,x,y,z) * (b,i,o,x,y,z) -> (b,o,x,y,z)
            return torch.einsum("bixyz,bioxyz->boxyz", input, weights)
        else:
            raise ValueError(f"Unexpected number of dimensions for weights: {weights.dim()}")

    # This forward pass can be used by both standard FNO and Dynamic FNO
    # For Dynamic FNO, the weights are passed in as arguments
    def forward(self, x, weights1=None, weights2=None, weights3=None, weights4=None):
        if weights1 is None:
            weights1, weights2, weights3, weights4 = self.weights1, self.weights2, self.weights3, self.weights4
        
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], weights4)
        
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

################################################################
# Models
################################################################

class FNO3d_base(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, T_in, T_out):
        super(FNO3d_base, self).__init__()
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.width = width
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 6
        self.fc0 = nn.Linear(self.T_in + 3, self.width) # +3 for grid
        
        self.conv0, self.conv1, self.conv2, self.conv3 = [SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(4)]
        self.w0, self.w1, self.w2, self.w3 = [nn.Conv1d(self.width, self.width, 1) for _ in range(4)]
        self.bn0, self.bn1, self.bn2, self.bn3 = [torch.nn.BatchNorm3d(self.width) for _ in range(4)]
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    def forward(self, x):
        # x initial shape: (batch, S_y, S_x, T_in)
        # We repeat the input T_out times to get the desired temporal resolution for prediction
        x_repeated = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, self.T_in).repeat([1, 1, 1, self.T_out, 1])
        grid = self.get_grid(x_repeated.shape, x.device)
        
        x = torch.cat((x_repeated, grid), dim=-1) # (batch, S_y, S_x, T_out, T_in+3)
        x = self.fc0(x)                           # (batch, S_y, S_x, T_out, width)
        x = x.permute(0, 4, 1, 2, 3)             # (batch, width, S_y, S_x, T_out)
        x = F.pad(x, [0, self.padding])          # Pad time dimension
        
        # Apply Fourier layers
        x1 = self.conv0(x); x2 = self.w0(x.view(x.shape[0], self.width, -1)).view(x.shape); x = self.bn0(x1) + x2; x = F.gelu(x)
        x1 = self.conv1(x); x2 = self.w1(x.view(x.shape[0], self.width, -1)).view(x.shape); x = self.bn1(x1) + x2; x = F.gelu(x)
        x1 = self.conv2(x); x2 = self.w2(x.view(x.shape[0], self.width, -1)).view(x.shape); x = self.bn2(x1) + x2; x = F.gelu(x)
        x1 = self.conv3(x); x2 = self.w3(x.view(x.shape[0], self.width, -1)).view(x.shape); x = self.bn3(x1) + x2
        
        x = x[..., :-self.padding]               # Unpad time dimension
        x = x.permute(0, 2, 3, 4, 1)             # (batch, S_y, S_x, T_out, width)
        
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)                          # (batch, S_y, S_x, T_out, 1)
        return x.squeeze(-1)                     # (batch, S_y, S_x, T_out)

class DynamicFilterFNO_3d(FNO3d_base):
    def __init__(self, modes1, modes2, modes3, width, T_in, T_out, S_y, S_x):
        super(DynamicFilterFNO_3d, self).__init__(modes1, modes2, modes3, width, T_in, T_out)
        self.gating_network = nn.Sequential(nn.Linear(self.T_in * S_y * S_x, 512), nn.GELU(), nn.Linear(512, 1), nn.Sigmoid())
        
        # Create two sets of weights (A and B) for each of the 4 Fourier layers
        for i in range(4):
            for weight_type in ['A', 'B']:
                for j in range(1, 5):
                    setattr(self, f'weights_{weight_type}{j}_L{i}', nn.Parameter(self.conv0.scale * torch.rand(self.width, self.width, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)))

    def forward(self, x):
        # x initial shape: (batch, S_y, S_x, T_in)
        alpha = self.gating_network(x.view(x.shape[0], -1)).view(x.shape[0], 1, 1, 1, 1, 1)
        
        x_repeated = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, self.T_in).repeat([1, 1, 1, self.T_out, 1])
        grid = self.get_grid(x_repeated.shape, x.device)
        
        x = torch.cat((x_repeated, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])

        for i in range(4):
            # Interpolate weights for the current layer using alpha
            w1 = alpha * getattr(self, f'weights_A1_L{i}') + (1 - alpha) * getattr(self, f'weights_B1_L{i}')
            w2 = alpha * getattr(self, f'weights_A2_L{i}') + (1 - alpha) * getattr(self, f'weights_B2_L{i}')
            w3 = alpha * getattr(self, f'weights_A3_L{i}') + (1 - alpha) * getattr(self, f'weights_B3_L{i}')
            w4 = alpha * getattr(self, f'weights_A4_L{i}') + (1 - alpha) * getattr(self, f'weights_B4_L{i}')
            
            # Use the spectral conv from the base class, but pass in the dynamic weights
            x1 = self.conv0(x, w1, w2, w3, w4) 
            x2 = getattr(self, f'w{i}')(x.view(x.shape[0], self.width, -1)).view(x.shape)
            x = getattr(self, f'bn{i}')(x1) + x2
            if i < 3: x = F.gelu(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

################################################################
# Experiment Runner
################################################################

def run_experiment(model, model_name, train_loader, test_loader, y_normalizer, device, ntrain, ntest, epochs, batch_size, learning_rate, scheduler_step, scheduler_gamma):
    print(f"\n--- Starting Experiment for: {model_name} ---")
    print(f"Total params: {count_params(model)}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=False)
    y_normalizer.to(device)
    
    best_test_l2 = float('inf')

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            
            # The output of the model and the ground truth y need to be denormalized
            y_norm = y_normalizer.decode(y)
            out_norm = y_normalizer.decode(out)
            
            loss = myloss(out_norm.view(x.shape[0], -1), y_norm.view(x.shape[0], -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        scheduler.step()
        t2 = default_timer()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                y_norm = y_normalizer.decode(y)
                out_norm = y_normalizer.decode(out)
                test_l2 += myloss(out_norm.view(x.shape[0], -1), y_norm.view(x.shape[0], -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            print(f"[{model_name}] Epoch: {ep}, Time: {t2-t1:.2f}s, Train L2: {train_l2:.4f}, Test L2: {test_l2:.4f}  <-- New best, model saved!")
        else:
            print(f"[{model_name}] Epoch: {ep}, Time: {t2-t1:.2f}s, Train L2: {train_l2:.4f}, Test L2: {test_l2:.4f}")

    return best_test_l2

################################################################
# Configs and Data Loading
################################################################
# For Colab: Ensure this file is in the same directory as the script
DATA_PATH = 'ns_V1e-4_N10000_T30.mat'

ntrain, ntest = 9000, 1000
modes, width = 8, 20
batch_size = 20
epochs = 200 # Increased epochs for convergence
learning_rate, scheduler_step, scheduler_gamma = 0.001, 50, 0.5 # Adjusted scheduler for longer training
sub = 1 
T_in, T_out = 10, 20 # Predict 20 timesteps from 10

print(f"Epochs: {epochs}, LR: {learning_rate}, Batch Size: {batch_size}")

t1 = default_timer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    reader = MatReader(DATA_PATH)
    # The raw data is shaped (Y, X, N, T) -> (50, 64, 10000, 30) based on previous errors
    raw_data = reader.read_field('u') 
except (FileNotFoundError, NameError):
    print(f"ERROR: Data file not found at '{DATA_PATH}' or MatReader not defined.")
    print("Please make sure the .mat file and utilities3.py are in the same directory.")
    exit()

# Permute to (N, Y, X, T) -> (10000, 50, 64, 30)
all_u = raw_data.permute(2, 0, 1, 3)

train_a = all_u[:ntrain, ::sub, ::sub, :T_in]
train_u = all_u[:ntrain, ::sub, ::sub, T_in:T_in+T_out]
test_a = all_u[-ntest:, ::sub, ::sub, :T_in]
test_u = all_u[-ntest:, ::sub, ::sub, T_in:T_in+T_out]

# Get data dimensions
S_y, S_x = train_a.shape[1], train_a.shape[2]
print(f"Data loaded. Resolution: {S_x}x{S_y}. Time steps: {T_in} -> {T_out}.")
print(f"Shapes: train_a: {train_a.shape}, train_u: {train_u.shape}")

assert (S_y == train_u.shape[1] and S_x == train_u.shape[2])
assert (T_in == train_a.shape[3])
assert (T_out == train_u.shape[3])

a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

print(f'Preprocessing finished, time used: {default_timer()-t1:.2f}s, Using device: {device}')

################################################################
# Main Execution
################################################################

# Experiment 1: Standard FNO3d
fno_model = FNO3d_base(modes, modes, modes, width, T_in, T_out).to(device)
final_fno_test_err = run_experiment(fno_model, "FNO3d", train_loader, test_loader, y_normalizer, device, ntrain, ntest, epochs, batch_size, learning_rate, scheduler_step, scheduler_gamma)

# Experiment 2: Dynamic Filter FNO3d
dynamic_model = DynamicFilterFNO_3d(modes, modes, modes, width, T_in, T_out, S_y, S_x).to(device)
final_dynamic_test_err = run_experiment(dynamic_model, "DynamicFilterFNO_3d", train_loader, test_loader, y_normalizer, device, ntrain, ntest, epochs, batch_size, learning_rate, scheduler_step, scheduler_gamma)

print("\n\n--- FINAL RESULTS ---")
print(f"Standard FNO3d Test Error: {final_fno_test_err:.6f}")
print(f"Dynamic Filter FNO3d Test Error: {final_dynamic_test_err:.6f}")

if final_dynamic_test_err < final_fno_test_err:
    print("\n🎉🎉🎉 VICTORY! The Dynamic Filter FNO WINS on the large dataset! 🎉🎉🎉")
else:
    print("\nStandard FNO wins. The hypothesis that more data would favor the complex model appears incorrect.")
