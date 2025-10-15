import numpy as np
import time
from scipy.interpolate import interp1d
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt
import fourier_ops_custom
from scipy.fft import fftfreq, ifft
import os

# Set up matplotlib to support Chinese characters if needed
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti TC', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# --- Global Configuration ---
# The script can run in two modes:
# "FINAL_COMPARISON": Compares CFT/ICFT vs FFT/IFFT for multiple signals.
# "SENSITIVITY_ANALYSIS": Analyzes the effect of L_inv on ICFT accuracy and runtime for a single signal.
OVERALL_RUN_MODE = "FINAL_COMPARISON"

# Define signal functions
def signal_linear_chirp(t):
    """A linear chirp signal."""
    return np.cos(2 * np.pi * (2 * t + 5 * t**2))

def gaussian_signal(t, sigma_val=0.5):
    """A Gaussian pulse centered at t=0."""
    return np.exp(-t**2 / (2 * sigma_val**2))

def signal_multi_sine(t):
    """A signal composed of two sine waves."""
    return (np.sin(2 * np.pi * 5 * t) + 0.7 * np.cos(2 * np.pi * 18 * t))

# --- Signal configurations for the final run ---
# Parameters are set for high-precision comparison.
signal_configurations = [
    {
        "name": "LinearChirp_Compare_Balanced",
        "func": signal_linear_chirp,
        "t_min": -0.5, "t_max": 2.0, "N_samples": 2048,
        "L_cft": 1000, "M_cft": 30, 
        "L_inv": 2000, "M_inv": 30, # Balanced high-precision and speed
        "u_range_cft": [-70.0, 70.0]
    },
    {
        "name": "GaussianPulse_Compare_Balanced",
        "func": gaussian_signal,
        "t_min": -2.0, "t_max": 2.0, "N_samples": 1024,
        "L_cft": 200, "M_cft": 40,
        "L_inv": 1000, "M_inv": 30, # Balanced high-precision and speed
        "u_range_cft": [-20.0, 20.0]
    },
    {
        "name": "MultiSine_Compare_Balanced",
        "func": signal_multi_sine,
        "t_min": -1.0, "t_max": 1.0, "N_samples": 2048,
        "L_cft": 800, "M_cft": 40,
        "L_inv": 2000, "M_inv": 30, # Balanced high-precision and speed
        "u_range_cft": [-40.0, 40.0]
    }
]

# This helper function is needed for the benchmark spectrum calculation.
def create_callable_spectrum_precomputed_quad_nodes(
    signal_func_to_integrate, t_min_integration, t_max_integration,
    quad_epsabs, quad_epsrel, quad_limit,
    freq_min, freq_max, L_segments, M_cheb,
    name_for_error="precomputed_quad_nodes_spectrum",
    verbose_level=0
):
    def _integrand_real_direct_nested(t, u_val_local, current_sig_func_local):
        return np.real(current_sig_func_local(t) * np.exp(-2j * np.pi * u_val_local * t))
    def _integrand_imag_direct_nested(t, u_val_local, current_sig_func_local):
        return np.imag(current_sig_func_local(t) * np.exp(-2j * np.pi * u_val_local * t))
    def _compute_spectrum_at_freq_direct_nested(u_val_single_ld, sig_func, t_min, t_max, q_epsabs, q_epsrel, q_limit):
        u_val_float = float(u_val_single_ld)
        real_part_ld, _ = scipy.integrate.quad(_integrand_real_direct_nested, t_min, t_max, args=(u_val_float, sig_func), epsabs=q_epsabs, epsrel=q_epsrel, limit=q_limit)
        imag_part_ld, _ = scipy.integrate.quad(_integrand_imag_direct_nested, t_min, t_max, args=(u_val_float, sig_func), epsabs=q_epsabs, epsrel=q_epsrel, limit=q_limit)
        return np.complex128(np.longdouble(real_part_ld) + 1j * np.longdouble(imag_part_ld))

    all_cheb_nodes = []
    segment_width = (np.longdouble(freq_max) - np.longdouble(freq_min)) / L_segments
    cheb_nodes_std_neg1_to_1 = np.polynomial.chebyshev.chebpts2(M_cheb + 1)
    for i_seg in range(L_segments):
        seg_f_min_ld = np.longdouble(freq_min) + i_seg * segment_width
        seg_f_max_ld = np.longdouble(freq_min) + (i_seg + 1) * segment_width
        current_segment_nodes_ld = cheb_nodes_std_neg1_to_1 * (seg_f_max_ld - seg_f_min_ld) / np.longdouble(2.0) + \
                                   (seg_f_max_ld + seg_f_min_ld) / np.longdouble(2.0)
        all_cheb_nodes.extend(current_segment_nodes_ld)
    unique_nodes_ld = np.array(sorted(list(set(all_cheb_nodes))), dtype=np.longdouble)
    
    precomputed_freqs_ld = unique_nodes_ld
    precomputed_spec_vals_complex = np.empty(len(unique_nodes_ld), dtype=np.complex128)

    for i, node_ld in enumerate(precomputed_freqs_ld):
        precomputed_spec_vals_complex[i] = _compute_spectrum_at_freq_direct_nested(
            node_ld, signal_func_to_integrate,
            np.longdouble(t_min_integration), np.longdouble(t_max_integration),
            quad_epsabs, quad_epsrel, quad_limit
        )
    
    interp_freqs = precomputed_freqs_ld.astype(np.float64)
    interp_spec_real = precomputed_spec_vals_complex.real.astype(np.float64)
    interp_spec_imag = precomputed_spec_vals_complex.imag.astype(np.float64)

    interp_func_real = interp1d(interp_freqs, interp_spec_real, kind='cubic', bounds_error=False, fill_value=0.0)
    interp_func_imag = interp1d(interp_freqs, interp_spec_imag, kind='cubic', bounds_error=False, fill_value=0.0)

    def get_interpolated_precomputed_spectrum_values(u_vals_array_input):
        u_vals_float64 = np.asarray(u_vals_array_input, dtype=np.float64)
        real_parts = interp_func_real(u_vals_float64)
        imag_parts = interp_func_imag(u_vals_float64)
        return (real_parts + 1j * imag_parts).astype(np.complex128)

    return get_interpolated_precomputed_spectrum_values

# --- Main Execution Logic ---
if __name__ == "__main__":
    
    plots_path = "plots"
    os.makedirs(plots_path, exist_ok=True)
    
    if OVERALL_RUN_MODE == "FINAL_COMPARISON":
        print(f"--- Running Final Comparison: CFT/ICFT vs. FFT/IFFT ---")
        print(f"Plots will be saved to '{plots_path}/'")

        # Define high-quality QUAD parameters for benchmark spectrum calculation
        QUAD_EPS_ABS = 1.49e-11
        QUAD_EPS_REL = 1.49e-11
        QUAD_LIMIT = 4000
        
        for sig_config in signal_configurations:
            sig_name = sig_config['name']
            print(f"\\nProcessing signal: {sig_name}...")

            # === 1. Setup Coordinates and Original Signal ===
            t_coords_sig = np.linspace(sig_config['t_min'], sig_config['t_max'], sig_config['N_samples'], dtype=np.longdouble)
            dt_sig = (sig_config['t_max'] - sig_config['t_min']) / (sig_config['N_samples'] - 1)
            original_signal = sig_config['func'](t_coords_sig)

            # === 2. PATH 1: Our High-Precision CFT -> ICFT pipeline ===
            
            # 2.1 Calculate High-Precision Continuous Spectrum (CFT)
            u_range_cft = sig_config.get('u_range_cft', [-70.0, 70.0])
            u_coords_cft = np.linspace(u_range_cft[0], u_range_cft[1], 4000, dtype=np.longdouble) # High-res freq coordinates for plotting

            cft_spec_callable = create_callable_spectrum_precomputed_quad_nodes(
                sig_config['func'], sig_config['t_min'], sig_config['t_max'],
                QUAD_EPS_ABS, QUAD_EPS_REL, QUAD_LIMIT,
                u_range_cft[0], u_range_cft[1], 
                sig_config['L_cft'], sig_config['M_cft'],
                name_for_error=f"CFT_{sig_name}"
            )
            cft_spectrum_vals = cft_spec_callable(u_coords_cft)
            
            # 2.2 Dynamically find effective frequency range for ICFT
            abs_spec = np.abs(cft_spectrum_vals)
            max_spec_val = np.max(abs_spec)
            significant_indices = np.where(abs_spec > max_spec_val * 1e-5)[0] # Stricter threshold
            
            if len(significant_indices) > 0:
                u_min_effective = u_coords_cft[significant_indices[0]]
                u_max_effective = u_coords_cft[significant_indices[-1]]
            else:
                u_min_effective, u_max_effective = u_range_cft

            # 2.3 Reconstruct signal using Our ICFT with the effective range
            our_icft_reconstruction, _ = fourier_ops_custom.compute_icft(
                spectrum_callable=cft_spec_callable,
                freq_min=u_min_effective, 
                freq_max=u_max_effective,
                t_values=t_coords_sig,
                L_segments=sig_config['L_inv'],
                M_cheb=sig_config['M_inv'],
                du_actual=0
            )

            # === 3. PATH 2: Standard FFT -> IFFT pipeline ===
            fft_spectrum = scipy.fft.fft(original_signal)
            fft_reconstruction = scipy.fft.ifft(fft_spectrum)
            fft_freqs = scipy.fft.fftfreq(sig_config['N_samples'], d=dt_sig)
            
            # === 4. Plotting All Comparisons ===
            plt.style.use('seaborn-v0_8-whitegrid')

            # --- Plot 1: Spectrum Comparison ---
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            # Plot our CFT
            ax1.plot(u_coords_cft, np.abs(cft_spectrum_vals), label='Our CFT (High-Precision)', color='red')
            # Plot FFT (shifted for standard visualization)
            shifted_fft_freqs = np.fft.fftshift(fft_freqs)
            shifted_fft_spectrum = np.fft.fftshift(fft_spectrum)
            # Normalizing FFT magnitude to be comparable to CFT. FFT magnitude scales with N.
            ax1.plot(shifted_fft_freqs, np.abs(shifted_fft_spectrum) * dt_sig, label='FFT', linestyle='--', color='blue', alpha=0.7)
            ax1.set_title(f"Spectrum Comparison - {sig_name}")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Spectrum Magnitude")
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlim(u_range_cft) # Use same freq limits for comparison
            plt.tight_layout()
            plt.savefig(os.path.join(plots_path, f"{sig_name}_spectrum_comparison.png"))
            plt.close(fig1)

            # --- Plot 2: Time Domain Reconstruction Comparison ---
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(t_coords_sig, original_signal, label='Original Signal', linewidth=2.5, color='black')
            ax2.plot(t_coords_sig, np.real(our_icft_reconstruction), label='Our ICFT Reconstruction', linestyle='--', color='red')
            ax2.plot(t_coords_sig, np.real(fft_reconstruction), label='FFT-IFFT Reconstruction', linestyle=':', color='blue')
            ax2.set_title(f"Time Domain Reconstruction - {sig_name}")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_path, f"{sig_name}_reconstruction_comparison.png"))
            plt.close(fig2)

            # --- Plot 3: Reconstruction Error Comparison ---
            error_our_icft = np.abs(np.real(our_icft_reconstruction) - original_signal)
            error_fft_ifft = np.abs(np.real(fft_reconstruction) - original_signal)
            mae_our_icft = np.mean(error_our_icft)
            mae_fft_ifft = np.mean(error_fft_ifft)

            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(t_coords_sig, error_our_icft, label=f'Our ICFT Error (MAE: {mae_our_icft:.2e})', color='red')
            ax3.plot(t_coords_sig, error_fft_ifft, label=f'FFT-IFFT Error (MAE: {mae_fft_ifft:.2e})', color='blue', linestyle='--')
            ax3.set_title(f"Reconstruction Absolute Error - {sig_name}")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Absolute Error")
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, which="both")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_path, f"{sig_name}_error_comparison.png"))
            plt.close(fig3)

        print("\\n\\nFinal comparison complete. All plots saved to 'plots/' directory.")

    elif OVERALL_RUN_MODE == "SENSITIVITY_ANALYSIS":
        print(f"--- Running Sensitivity Analysis for L_inv ---")
        
        # 1. Select a single signal for the analysis
        sig_config = signal_configurations[0] # Using LinearChirp_Compare
        sig_name = sig_config['name']
        print(f"Using signal: {sig_name}")

        # 2. Define the parameter range to test
        L_inv_values = [500, 1000, 2000, 4000, 8000, 12000]
        mae_results = []
        runtime_results = []

        # 3. Perform pre-computation outside the loop (Calculate the "true" spectrum once)
        print("Pre-calculating high-precision spectrum...")
        t_coords_sig = np.linspace(sig_config['t_min'], sig_config['t_max'], sig_config['N_samples'], dtype=np.longdouble)
        original_signal = sig_config['func'](t_coords_sig)
        
        QUAD_EPS_ABS = 1.49e-11
        QUAD_EPS_REL = 1.49e-11
        QUAD_LIMIT = 4000
        u_range_cft = sig_config.get('u_range_cft')
        
        cft_spec_callable = create_callable_spectrum_precomputed_quad_nodes(
            sig_config['func'], sig_config['t_min'], sig_config['t_max'],
            QUAD_EPS_ABS, QUAD_EPS_REL, QUAD_LIMIT,
            u_range_cft[0], u_range_cft[1], 
            sig_config['L_cft'], sig_config['M_cft']
        )
        
        # Dynamically find effective frequency range for ICFT once
        u_coords_cft_for_range = np.linspace(u_range_cft[0], u_range_cft[1], 4000, dtype=np.longdouble)
        cft_spectrum_vals_for_range = cft_spec_callable(u_coords_cft_for_range)
        abs_spec = np.abs(cft_spectrum_vals_for_range)
        max_spec_val = np.max(abs_spec)
        significant_indices = np.where(abs_spec > max_spec_val * 1e-5)[0]
        u_min_effective, u_max_effective = (u_coords_cft_for_range[significant_indices[0]], u_coords_cft_for_range[significant_indices[-1]]) if len(significant_indices) > 0 else u_range_cft
        print(f"Effective frequency range for analysis: [{u_min_effective:.2f}, {u_max_effective:.2f}]")

        # 4. Loop through the parameter values and record metrics
        for l_inv in L_inv_values:
            print(f"\\nTesting L_inv = {l_inv}...")
            
            start_time = time.time()
            
            our_icft_reconstruction, _ = fourier_ops_custom.compute_icft(
                spectrum_callable=cft_spec_callable,
                freq_min=u_min_effective, 
                freq_max=u_max_effective,
                t_values=t_coords_sig,
                L_segments=l_inv,
                M_cheb=sig_config['M_inv'], # Keep M_inv fixed
                du_actual=0
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            error = np.abs(np.real(our_icft_reconstruction) - original_signal)
            mae = np.mean(error)
            
            mae_results.append(mae)
            runtime_results.append(runtime)
            
            print(f"  -> MAE: {mae:.4e}, Runtime: {runtime:.4f} s")

        # 5. Plot the results
        print("\\nGenerating sensitivity analysis plots...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot 1: MAE vs. L_inv
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(L_inv_values, mae_results, 'o-', color='blue')
        ax1.set_xlabel("Number of Segments (L_inv)")
        ax1.set_ylabel("Mean Absolute Error (MAE)")
        ax1.set_title(f"ICFT Accuracy vs. L_inv for {sig_name}")
        ax1.set_yscale('log')
        ax1.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f"{sig_name}_sensitivity_mae_vs_L.png"))
        plt.close(fig1)

        # Plot 2: Runtime vs. L_inv
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(L_inv_values, runtime_results, 's-', color='green')
        ax2.set_xlabel("Number of Segments (L_inv)")
        ax2.set_ylabel("Runtime (seconds)")
        ax2.set_title(f"ICFT Runtime vs. L_inv for {sig_name}")
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f"{sig_name}_sensitivity_runtime_vs_L.png"))
        plt.close(fig2)
        
        print(f"\\nSensitivity analysis complete. Plots saved to '{plots_path}/'")
        print("--- Results Summary ---")
        print(f"{'L_inv':>10} | {'MAE':>15} | {'Runtime (s)':>15}")
        print("-" * 45)
        for i, l_inv in enumerate(L_inv_values):
            print(f"{l_inv:>10} | {mae_results[i]:>15.4e} | {runtime_results[i]:>15.4f}")
    
    else:
        print(f"Unknown OVERALL_RUN_MODE: '{OVERALL_RUN_MODE}'. Please set to 'FINAL_COMPARISON' or 'SENSITIVITY_ANALYSIS'.") 