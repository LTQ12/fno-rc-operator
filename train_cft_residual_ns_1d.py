import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from timeit import default_timer
import h5py

from fourier_1d_cft_residual import FNO_RC_1D
from utilities3 import LpLoss, UnitGaussianNormalizer, count_params
from Adam import Adam

def create_training_data_from_sequence(sequence, T_in, T_out, step=1):
    """
    Creates training pairs (x, y) from a long time sequence.
    """
    N, S, T = sequence.shape
    num_samples = (T - T_in - T_out) // step
    
    X = torch.zeros(N * num_samples, S, T_in)
    Y = torch.zeros(N * num_samples, S, T_out)
    
    idx = 0
    for i in range(N):
        for t in range(0, T - T_in - T_out, step):
            X[idx] = sequence[i, :, t : t + T_in]
            Y[idx] = sequence[i, :, t + T_in : t + T_in + T_out]
            idx += 1
            
    return X, Y


def main(args):
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    # --- Data Loading ---
    try:
        with h5py.File(args.data_path, 'r') as f:
            print(f"DEBUG: h5py 'output' key initial shape: {f['output'].shape}")
            # The shape from h5py is (S, T, N). We need to permute it to (N, S, T) for the model.
            # This is the definitive fix.
            u_field = torch.from_numpy(f['output'][:].astype(np.float32)).permute(2, 0, 1)
            print(f"DEBUG: Shape of u_field after correct permute: {u_field.shape}")
    except Exception as e:
        print(f"Error: Data file not found or could not be read at '{args.data_path}'")
        print(e)
        return

    # Create sequence-to-sequence pairs from the long simulation
    train_a, train_y = create_training_data_from_sequence(u_field, args.T_in, args.T_out)
    
    ntrain = int(train_a.shape[0] * 0.8)
    ntest = train_a.shape[0] - ntrain

    shuffled_indices = torch.randperm(train_a.shape[0])
    train_indices = shuffled_indices[:ntrain]
    test_indices = shuffled_indices[-ntest:]

    test_a = train_a[test_indices]
    test_y = train_y[test_indices]
    train_a = train_a[train_indices]
    train_y = train_y[train_indices]

    print(f"DEBUG: Shape of final train_a for model input: {train_a.shape}")
    print(f"Data shapes: train_a: {train_a.shape}, train_y: {train_y.shape}")

    # --- Normalization ---
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_y)
    train_y = y_normalizer.encode(train_y)
    y_normalizer.to(device)

    # --- DataLoader ---
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_y),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_y),
        batch_size=args.batch_size,
        shuffle=False
    )

    # --- Model Initialization ---
    model = FNO_RC_1D(args.modes, args.width, in_channels=args.T_in, out_channels=args.T_out).to(device)
    print(f"Model: FNO-RC 1D")
    print(f"Parameters: {count_params(model)}")
    print(f"Hyperparameters: Modes={args.modes}, Width={args.width}, LR={args.learning_rate}")

    # --- Training ---
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_func = LpLoss(size_average=False)

    print("--- Starting Training ---")
    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            
            loss = loss_func(y_normalizer.decode(out), y_normalizer.decode(y))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
        
        scheduler.step()
        train_l2 /= ntrain

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += loss_func(y_normalizer.decode(out), y_normalizer.decode(y)).item()
        
        test_l2 /= ntest
        t2 = default_timer()
        
        if (ep + 1) % args.log_interval == 0:
            print(f"Epoch {ep+1}/{args.epochs} | Time: {t2-t1:.2f}s | Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f}")

    # --- Save Model ---
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_save_dir, 'fno_rc_1d.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"--- Training Finished ---\nModel saved to {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 1D FNO-RC')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/burgers.mat', help='Path to the .mat data file in Google Drive')
    parser.add_argument('--modes', type=int, default=16, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=64, help='Feature width of the model')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--log_interval', type=int, default=10, help='Epoch interval for logging')
    parser.add_argument('--model_save_dir', type=str, default='models', help='Directory to save the trained model')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cuda:0" or "cpu")')
    
    args = parser.parse_args()
    main(args) 