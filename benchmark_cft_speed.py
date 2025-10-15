"""
Benchmark for `torch.compile` on the Vectorized CFT Operator.

This script measures and compares the execution speed of the original
`vectorized_batched_cft` function against its `torch.compile`'d version
to quantify the performance gains from JIT compilation.
"""
import torch
from timeit import default_timer as timer
import numpy as np

# Make sure we have the correct operator
from fourier_ops_vectorized import vectorized_batched_cft, vectorized_batched_cft_decomposed

def benchmark():
    """Runs the benchmark and prints the results."""
    
    # --- 1. Setup & Sanity Check ---
    print(f"PyTorch Version: {torch.__version__}")
    if not hasattr(torch, 'compile'):
        print("`torch.compile` is not available in this PyTorch version. Please upgrade to PyTorch 2.0+.")
        return
    
    # --- 2. Create Dummy Data ---
    # Realistic parameters from our previous experiments
    batch_size = 20
    in_channels = 64
    n_samples = 1024 # s = 2**13 // 2**3
    
    # Use CPU for a fair comparison on this machine
    device = torch.device('cpu') 

    # Generate some random signal data
    x = torch.randn(batch_size, in_channels, n_samples, device=device)
    t_coords = torch.linspace(0, 1, n_samples, device=device)
    f_points = torch.fft.fftfreq(n_samples, d=1/n_samples).to(device)
    
    # Optimal parameters we found
    L = 10
    M = 8

    print("\n--- Benchmark Setup ---")
    print(f"Device: {device}")
    print(f"Data Shape (batch, channels, samples): ({batch_size}, {in_channels}, {n_samples})")
    print(f"CFT Parameters: L={L}, M={M}")
    print("-------------------------\n")

    # --- 3. Compile the Models ---
    print("Compiling operators with `torch.compile`...")
    try:
        # Original complex version (expected to have poor/no speedup)
        compiled_cft_original = torch.compile(vectorized_batched_cft)
        
        # New real-decomposed version (expected to have significant speedup)
        compiled_cft_decomposed = torch.compile(vectorized_batched_cft_decomposed)
        print("Compilation successful.")
    except Exception as e:
        print(f"Could not compile the function. Error: {e}")
        return

    # --- 4. Warm-up Runs ---
    print("Warming up all three function versions...")
    _ = vectorized_batched_cft(x, t_coords, f_points, L, M)
    _ = vectorized_batched_cft_decomposed(x, t_coords, f_points, L, M)
    _ = compiled_cft_decomposed(x, t_coords, f_points, L, M)
    # No need to warm up the original compiled version as it falls back to eager
    print("Warm-up complete.")

    # --- 5. Run Benchmark ---
    n_runs = 20
    print(f"\nRunning benchmark ({n_runs} iterations)...")

    # Version 1: Original (Complex)
    start_time = timer()
    for _ in range(n_runs):
        _ = vectorized_batched_cft(x, t_coords, f_points, L, M)
    end_time = timer()
    original_time = (end_time - start_time) / n_runs

    # Version 2: Decomposed (Real, Eager)
    start_time = timer()
    for _ in range(n_runs):
        _ = vectorized_batched_cft_decomposed(x, t_coords, f_points, L, M)
    end_time = timer()
    decomposed_eager_time = (end_time - start_time) / n_runs

    # Version 3: Decomposed (Real, Compiled)
    start_time = timer()
    for _ in range(n_runs):
        _ = compiled_cft_decomposed(x, t_coords, f_points, L, M)
    end_time = timer()
    decomposed_compiled_time = (end_time - start_time) / n_runs

    # --- 6. Report Results ---
    speedup_vs_original = original_time / decomposed_compiled_time
    speedup_vs_eager_decomposed = decomposed_eager_time / decomposed_compiled_time
    
    print("\n--- Benchmark Results ---")
    print(f"1. Original (Complex) Avg Time:      {original_time:.4f} seconds")
    print(f"2. Decomposed (Real, Eager) Avg Time:{decomposed_eager_time:.4f} seconds")
    print(f"3. Decomposed (Real, Compiled) Avg Time:{decomposed_compiled_time:.4f} seconds")
    print("---------------------------------")
    print(f"Speed-up (Compiled Decomposed vs. Original): {speedup_vs_original:.2f}x")
    print(f"Speed-up (Compile Effect on Decomposed):     {speedup_vs_eager_decomposed:.2f}x")
    print("=================================\n")

if __name__ == "__main__":
    benchmark() 