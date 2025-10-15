import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import argparse
from timeit import default_timer

# Import the optimized FNO with CFT/ICFT capabilities
from fourier_1d_optimized import FNO1dOptimized as FNO1d_CFT
from fourier_1d_gated import GatedFNO1d # Import our new Gated model

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ###############################################################
# # Helper: Data Loading
# ###############################################################
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
        return x

# ###############################################################
# # Helper: Loss Function
# ###############################################################
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.p = p
        self.d = d
        self.size_average = size_average
        self.reduction = reduction

    def rel(self, x, y):
        num = torch.norm(x.reshape(x.shape[0], -1) - y.reshape(y.shape[0], -1), self.p, dim=1)
        den = torch.norm(y.reshape(y.shape[0], -1), self.p, dim=1)
        return num / den

    def __call__(self, x, y):
        loss = self.rel(x, y)
        if self.reduction:
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)
        return loss

# ###############################################################
# # Helper: FNO (FFT-based) Model Definition
# ###############################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d_FFT(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d_FFT, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
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

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ###############################################################
# # Training and Evaluation Logic
# ###############################################################
def train_model(model_name, model, config, train_loader, test_loader, device):
    """
    Encapsulates the training and evaluation loop for a given model.
    """
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_func = LpLoss(size_average=False)

    train_losses = []
    test_losses = []
    
    print(f"\n--- Training {model_name} for {epochs} epochs ---")
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        train_l2 /= len(train_loader.dataset)
        train_losses.append(train_l2)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += loss_func(out.view(x.shape[0], -1), y.view(x.shape[0], -1)).item()
        
        test_l2 /= len(test_loader.dataset)
        test_losses.append(test_l2)
        
        t2 = default_timer()
        if (ep + 1) % 1 == 0:
             print(f"Epoch: {ep+1}/{epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.4f} | Test L2: {test_l2:.4f}")
    
    # Final evaluation to get predictions
    predictions = torch.zeros(len(test_loader.dataset), config['s'])
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze()
            predictions[i*config['batch_size']:(i+1)*config['batch_size']] = out

    return {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "predictions": predictions.cpu().numpy()
    }

# ###############################################################
# # Plotting Logic
# ###############################################################
def generate_comparison_plots(results_dict, y_test, s):
    """
    Generates and saves comparison plots for different FNO models.
    """
    # Get the number of epochs from the first model's results
    first_model_results = next(iter(results_dict.values()))
    epochs = len(first_model_results['train_loss'])
    epoch_range = range(1, epochs + 1)
    
    colors = {'FFT': 'b', 'Gated': 'r'}
    
    # Plot 1: Loss Curves
    plt.figure(figsize=(12, 7))
    for model_name, results in results_dict.items():
        color = colors.get(model_name, 'k') # Default to black if name not in map
        plt.plot(epoch_range, results['train_loss'], color=color, linestyle='-', label=f'{model_name} Train Loss')
        plt.plot(epoch_range, results['test_loss'], color=color, linestyle='--', label=f'{model_name} Test Loss')
    plt.title('Training and Test Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('L2 Relative Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('comparison_loss_curves.png')
    plt.show()

    # Plot 2: Example Predictions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    indices = [0, 25, 50, 75] # Example indices from the test set
    x_grid = np.linspace(0, 1, s)

    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        ax.plot(x_grid, y_test[idx], 'k-', label='Ground Truth', linewidth=2)
        for model_name, results in results_dict.items():
            color = colors.get(model_name, 'k')
            ax.plot(x_grid, results['predictions'][idx], color=color, linestyle='--', label=f'{model_name} Prediction')
        ax.set_title(f'Test Sample #{idx}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t=1)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_predictions.png')
    plt.show()
    
    # Plot 3: Error distribution
    plt.figure(figsize=(12, 6))
    for model_name, results in results_dict.items():
        error = np.abs(results['predictions'] - y_test)
        color = colors.get(model_name, 'k')
        plt.hist(error.flatten(), bins=50, alpha=0.7, label=f'{model_name} Error', color=color, density=True)
    plt.title('Distribution of Point-wise Absolute Error')
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_error_distribution.png')
    plt.show()


# ###############################################################
# # Main Execution Logic
# ###############################################################
def main(args):
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # --- Config ---
    config = {
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'modes': 16,
        'width': 64,
        'batch_size': 20,
        'subsampling_rate': 2**3,
    }
    config['s'] = 2**13 // config['subsampling_rate']

    # --- Data Loading ---
    print("\nLoading data...")
    try:
        reader = MatReader('burgers_data_R10.mat')
        x_data = reader.read_field('a')[:, ::config['subsampling_rate']]
        y_data = reader.read_field('u')[:, ::config['subsampling_rate']]
    except FileNotFoundError:
        print("Error: 'burgers_data_R10.mat' not found. Please ensure it's in the correct directory.")
        return

    ntrain = 1000
    ntest = 100
    x_train = x_data[:ntrain, :].reshape(ntrain, config['s'], 1)
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :].reshape(ntest, config['s'], 1)
    y_test = y_data[-ntest:, :]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=config['batch_size'], shuffle=False)
    print("Data loaded successfully.")

    # --- Model Training ---
    results = {}

    # --- Train FFT Model ---
    if 'fft' in args.model_types:
        model_fft = FNO1d_FFT(config['modes'], config['width']).to(device)
        print(f"FFT Model Parameters: {count_params(model_fft)}")
        results['FFT'] = train_model('FFT FNO', model_fft, config, train_loader, test_loader, device)

    # --- Train Gated Model ---
    if 'gated' in args.model_types:
        gated_params = {
            'cft_L_segments': args.cft_L,
            'cft_M_cheb': args.cft_M,
        }
        model_gated = GatedFNO1d(config['modes'], config['width'], **gated_params).to(device)
        print(f"\nGated FNO Model Parameters: {count_params(model_gated)}")
        print(f"Gating CFT Config: L={gated_params['cft_L_segments']}, M={gated_params['cft_M_cheb']}")
        results['Gated'] = train_model('Gated FNO', model_gated, config, train_loader, test_loader, device)
        
    # --- Plotting ---
    if len(results) > 1:
        print("\n--- Generating Comparison Plots ---")
        generate_comparison_plots(results, y_test.numpy(), config['s'])
        print("Plots saved to disk.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and Compare FNO models for 1D Burgers Equation.')
    
    parser.add_argument('--model_types', type=str, nargs='+', default=['fft', 'gated'], 
                        choices=['fft', 'gated'],
                        help="A list of models to train and compare.")
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs.')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Optimizer learning rate.')

    # Add specific arguments for CFT to allow for easy tuning
    parser.add_argument('--cft_L', type=int, default=20, help='Number of segments for CFT integration (L).')
    parser.add_argument('--cft_M', type=int, default=8, help='Chebyshev interpolation order for CFT (M).')

    args = parser.parse_args()
    main(args) 