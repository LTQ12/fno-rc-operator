import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
from scipy.interpolate import interp1d

# Import functions from cft_direct
from cft_direct import cft as cft_direct_func
from cft_direct import cft_inverse as cft_inverse_direct_func
from cft_direct import analytical_gaussian_fourier_transform

# --- 1. Setup ---
print("--- 1. Setup ---")
sigma = 0.5
def gaussian_signal(t):
    # Ensure t is float for calculations, np.exp expects float or array of floats
    t_float = np.asarray(t, dtype=float)
    return np.exp(-t_float**2 / (2 * sigma**2))

# Time and Frequency domain setup
t_min, t_max = -2.5, 2.5  # Approx 5*sigma
N_POINTS = 512
# Use np.longdouble for precision where it matters, but ensure compatibility with functions
t_coords = np.linspace(t_min, t_max, N_POINTS, endpoint=False, dtype=np.longdouble)
dt = (t_max - t_min) / N_POINTS
# Original signal, cast to complex128 for consistent comparison types later
f_original = gaussian_signal(t_coords).astype(np.complex128) 

u_coords_fft = fftfreq(N_POINTS, d=dt) # Returns float64
sort_indices = np.argsort(u_coords_fft)
# u_coords_sorted will be used as input for CFT, ensure it's float or longdouble as needed
u_coords_sorted = u_coords_fft[sort_indices].astype(np.longdouble)

# CFT/ICFT parameters
L_param_cft = 20  # For forward CFT
M_param_cft = 7   # For forward CFT

L_param_icft_optimized = 200 # Optimized L for ICFT
M_param_icft_optimized = 7   # Optimized M for ICFT

interp_kind = 'cubic'

print(f"Sigma: {sigma}")
print(f"Time interval: [{t_min}, {t_max}]")
print(f"N_POINTS: {N_POINTS}, dt: {dt:.4e}")
print(f"Frequency range (sorted): [{u_coords_sorted.min():.2f}, {u_coords_sorted.max():.2f}] Hz")
print(f"Forward CFT L: {L_param_cft}, M: {M_param_cft}")
print(f"Optimized ICFT L: {L_param_icft_optimized}, M: {M_param_icft_optimized}, Spectrum Interp: '{interp_kind}'")
print("\n--- End Setup ---\n")

# --- 2. Experiment 1: Forward CFT Error (Error_Forward) ---
print("--- 2. Experiment 1: Forward CFT Error ---")
start_time = time.time()
# cft_direct_func expects a callable that can handle np.longdouble if t_points_std is longdouble
# and u_values can be longdouble. It returns clongdouble.
F_cft_clongdouble = cft_direct_func(gaussian_signal, t_min, t_max, u_coords_sorted, L_param_cft, M_param_cft)
# Cast to complex128 for MAE calculation if F_ana is complex128
F_cft_complex = F_cft_clongdouble.astype(np.complex128)
end_time = time.time()
print(f"Forward CFT calculation time: {end_time - start_time:.4f}s")

# analytical_gaussian_fourier_transform returns float, cast to complex128
F_ana_complex = analytical_gaussian_fourier_transform(u_coords_sorted.astype(float), sigma).astype(np.complex128)

MAE_forward_real = np.mean(np.abs(np.real(F_cft_complex) - np.real(F_ana_complex)))
MAE_forward_imag = np.mean(np.abs(np.imag(F_cft_complex) - np.imag(F_ana_complex)))
MAE_forward_abs = np.mean(np.abs(np.abs(F_cft_complex) - np.abs(F_ana_complex)))

print(f"  MAE Forward CFT (Real part vs Analytical): {MAE_forward_real:.6e}")
print(f"  MAE Forward CFT (Imag part vs Analytical): {MAE_forward_imag:.6e}")
print(f"  MAE Forward CFT (Magnitude vs Analytical): {MAE_forward_abs:.6e}")
print("\n--- End Experiment 1 ---\n")

# --- 3. Experiment 2: ICFT Intrinsic Error (Error_ICFT_Intrinsic) ---
print("--- 3. Experiment 2: ICFT Intrinsic Error (Perfect Spectrum Input) ---")
def spectrum_callable_analytical(u_vals_in):
    # Ensure u_vals_in is float for analytical_gaussian_fourier_transform
    u_vals_float = np.asarray(u_vals_in, dtype=float)
    # Return complex128 as cft_inverse will cast its result to complex128
    return analytical_gaussian_fourier_transform(u_vals_float, sigma).astype(np.complex128)

start_time = time.time()
# cft_inverse_direct_func input t_values is longdouble.
# It returns complex128.
f_reconstructed_ideal_spec = cft_inverse_direct_func(
    spectrum_callable_analytical,
    u_coords_sorted.min(), # freq_min
    u_coords_sorted.max(), # freq_max
    t_coords,             # t_values
    L=L_param_icft_optimized, # Use optimized L for ICFT
    M=M_param_icft_optimized, # Use optimized M for ICFT
    du_actual=dt
)
end_time = time.time()
print(f"ICFT (from Analytical Spectrum, L={L_param_icft_optimized}, M={M_param_icft_optimized}) calculation time: {end_time - start_time:.4f}s")

# f_original is complex128. f_reconstructed_ideal_spec is complex128.
MAE_icft_intrinsic_real = np.mean(np.abs(np.real(f_reconstructed_ideal_spec) - np.real(f_original)))
MAE_icft_intrinsic_imag = np.mean(np.abs(np.imag(f_reconstructed_ideal_spec) - np.imag(f_original)))

print(f"  MAE ICFT Intrinsic (Real part vs Original): {MAE_icft_intrinsic_real:.6e}")
print(f"  MAE ICFT Intrinsic (Imag part vs Original): {MAE_icft_intrinsic_imag:.6e}")
print("\n--- End Experiment 2 ---\n")

# --- 4. Experiment 3: Total End-to-End Error (Error_Total) ---
print("--- 4. Experiment 3: Total End-to-End Error (CFT -> Interpolation -> ICFT) ---")
# F_cft_complex is from Experiment 1, type complex128.
# interp1d x and y inputs should be float for typical use.
interp_real_from_cft = interp1d(u_coords_sorted.astype(float), np.real(F_cft_complex).astype(float), kind=interp_kind, fill_value="extrapolate", bounds_error=False)
interp_imag_from_cft = interp1d(u_coords_sorted.astype(float), np.imag(F_cft_complex).astype(float), kind=interp_kind, fill_value="extrapolate", bounds_error=False)

def spec_callable_from_cft_combined(u_vals_in):
    u_vals_float = np.asarray(u_vals_in, dtype=float)
    # Returns complex (effectively complex128 due to float inputs to interp)
    return interp_real_from_cft(u_vals_float) + 1j * interp_imag_from_cft(u_vals_float)

start_time = time.time()
f_reconstructed_cft_spec = cft_inverse_direct_func(
    spec_callable_from_cft_combined,
    u_coords_sorted.min(), 
    u_coords_sorted.max(),
    t_coords,
    L=L_param_icft_optimized, # Use optimized L for ICFT
    M=M_param_icft_optimized, # Use optimized M for ICFT
    du_actual=dt
) # Returns complex128
end_time = time.time()
print(f"ICFT (from Interpolated CFT Spectrum, L={L_param_icft_optimized}, M={M_param_icft_optimized}) calculation time: {end_time - start_time:.4f}s")

MAE_total_real = np.mean(np.abs(np.real(f_reconstructed_cft_spec) - np.real(f_original)))
MAE_total_imag = np.mean(np.abs(np.imag(f_reconstructed_cft_spec) - np.imag(f_original)))

print(f"  MAE Total End-to-End (Real part vs Original): {MAE_total_real:.6e}")
print(f"  MAE Total End-to-End (Imag part vs Original): {MAE_total_imag:.6e}")
print("\n--- End Experiment 3 ---\n")

# --- 5. Analysis of Error Contributions (Qualitative Summary from MAEs) ---
print("--- 5. Summary of MAE values ---")
print(f"  Forward CFT Error (Real Part vs Analytical):      {MAE_forward_real:.6e}")
print(f"  Forward CFT Error (Imag Part vs Analytical):      {MAE_forward_imag:.6e}") # Effectively CFT's imag part
print(f"  Forward CFT Error (Magnitude vs Analytical):      {MAE_forward_abs:.6e}")
print(f"  ----------------------------------------------------")
print(f"  ICFT Intrinsic Error (Real Part vs Original):   {MAE_icft_intrinsic_real:.6e}")
print(f"  ICFT Intrinsic Error (Imag Part vs Original):   {MAE_icft_intrinsic_imag:.6e}") # Effectively recon's imag part
print(f"  ----------------------------------------------------")
print(f"  Total End-to-End Error (Real Part vs Original): {MAE_total_real:.6e}")
print(f"  Total End-to-End Error (Imag Part vs Original): {MAE_total_imag:.6e}") # Effectively recon's imag part
print("\n--- End Analysis ---\n")

# --- Plotting for visual inspection ---
# Plot 1: Forward CFT vs Analytical
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(u_coords_sorted.astype(float), np.real(F_ana_complex).astype(float), 'k--', label='Analytical Spec (Real)')
plt.plot(u_coords_sorted.astype(float), np.real(F_cft_complex).astype(float), 'b-', alpha=0.7, label=f'CFT Spec (Real) (L={L_param_cft},M={M_param_cft})')
plt.xlabel('Frequency (u)')
plt.ylabel('Real Part')
plt.title('Forward CFT: Real Parts')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(u_coords_sorted.astype(float), np.imag(F_ana_complex).astype(float), 'k--', label='Analytical Spec (Imag)')
plt.plot(u_coords_sorted.astype(float), np.imag(F_cft_complex).astype(float), 'r:', alpha=0.7, label=f'CFT Spec (Imag) (L={L_param_cft},M={M_param_cft})')
plt.xlabel('Frequency (u)')
plt.ylabel('Imaginary Part')
plt.title('Forward CFT: Imaginary Parts')
plt.legend()
plt.grid(True)
plt.suptitle(f'Exp 1: Forward CFT vs Analytical (L={L_param_cft},M={M_param_cft})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("error_decomp_exp1_forward_cft.png")
plt.close()
print("Saved plot for Experiment 1 to error_decomp_exp1_forward_cft.png")

# Plot 2: ICFT from Analytical Spectrum vs Original Signal
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(t_coords.astype(float), np.real(f_original).astype(float), 'k--', label='Original Signal (Real)')
plt.plot(t_coords.astype(float), np.real(f_reconstructed_ideal_spec).astype(float), 'b-', alpha=0.7, label=f'ICFT from Analytical (Real) (L={L_param_icft_optimized},M={M_param_icft_optimized})')
plt.xlabel('Time (t)')
plt.ylabel('Real Part')
plt.title('ICFT (Analytical Spec): Real Parts')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_coords.astype(float), np.imag(f_original).astype(float), 'k--', label='Original Signal (Imag)')
plt.plot(t_coords.astype(float), np.imag(f_reconstructed_ideal_spec).astype(float), 'r:', alpha=0.7, label=f'ICFT from Analytical (Imag) (L={L_param_icft_optimized},M={M_param_icft_optimized})')
plt.xlabel('Time (t)')
plt.ylabel('Imaginary Part')
plt.title('ICFT (Analytical Spec): Imaginary Parts')
plt.legend()
plt.grid(True)
plt.suptitle(f'Exp 2: ICFT from Analytical Spectrum (L={L_param_icft_optimized},M={M_param_icft_optimized})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("error_decomp_exp2_icft_intrinsic.png")
plt.close()
print("Saved plot for Experiment 2 to error_decomp_exp2_icft_intrinsic.png")

# Plot 3: End-to-End ICFT vs Original Signal
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(t_coords.astype(float), np.real(f_original).astype(float), 'k--', label='Original Signal (Real)')
plt.plot(t_coords.astype(float), np.real(f_reconstructed_cft_spec).astype(float), 'b-', alpha=0.7, label=f'ICFT from CFT Spec (Real) (L={L_param_icft_optimized},M={M_param_icft_optimized})')
plt.xlabel('Time (t)')
plt.ylabel('Real Part')
plt.title('End-to-End ICFT: Real Parts')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_coords.astype(float), np.imag(f_original).astype(float), 'k--', label='Original Signal (Imag)')
plt.plot(t_coords.astype(float), np.imag(f_reconstructed_cft_spec).astype(float), 'r:', alpha=0.7, label=f'ICFT from CFT Spec (Imag) (L={L_param_icft_optimized},M={M_param_icft_optimized})')
plt.xlabel('Time (t)')
plt.ylabel('Imaginary Part')
plt.title('End-to-End ICFT: Imaginary Parts')
plt.legend()
plt.grid(True)
plt.suptitle(f'Exp 3: End-to-End ICFT Reconstruction (L={L_param_icft_optimized},M={M_param_icft_optimized})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("error_decomp_exp3_total_error.png")
plt.close()
print("Saved plot for Experiment 3 to error_decomp_exp3_total_error.png")

# --- 6. Detailed Error Visualization for Amplification Analysis ---
print("\n--- 6. Detailed Error Visualization ---")

# 6.1 Spectrum Error delta_F(u)
delta_F_real = np.real(F_cft_complex) - np.real(F_ana_complex)
delta_F_imag = np.imag(F_cft_complex) - np.imag(F_ana_complex) # F_ana_complex is real, so this is imag(F_cft_complex)
delta_F_abs_diff = np.abs(F_cft_complex) - np.abs(F_ana_complex) # Difference of magnitudes

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(u_coords_sorted.astype(float), delta_F_real.astype(float), label='Real Part Error (F_cft - F_ana)')
plt.title('Spectrum Error: Real Part')
plt.xlabel('Frequency (u)')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(u_coords_sorted.astype(float), delta_F_imag.astype(float), label='Imag Part Error (F_cft - F_ana)')
plt.title('Spectrum Error: Imaginary Part')
plt.xlabel('Frequency (u)')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(u_coords_sorted.astype(float), delta_F_abs_diff.astype(float), label='Magnitude Error (|F_cft| - |F_ana|)')
plt.title('Spectrum Error: Magnitude Difference')
plt.xlabel('Frequency (u)')
plt.grid(True)
plt.legend()

plt.suptitle(f'Detailed Spectrum Error delta_F(u) (Forward CFT L={L_param_cft}, M={M_param_cft})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("error_viz_delta_F.png")
plt.close()
print("Saved plot for delta_F(u) to error_viz_delta_F.png")

# 6.2 Time Domain Error due to Imperfect Spectrum delta_f_due_to_F_err(t)
# f_total_reconstruction is f_reconstructed_cft_spec (from Exp3)
# f_ideal_reconstruction is f_reconstructed_ideal_spec (from Exp2)
delta_f_real_due_to_F_err = np.real(f_reconstructed_cft_spec) - np.real(f_reconstructed_ideal_spec)
delta_f_imag_due_to_F_err = np.imag(f_reconstructed_cft_spec) - np.imag(f_reconstructed_ideal_spec)

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(t_coords.astype(float), delta_f_real_due_to_F_err.astype(float), label='Real Part (f_total_recon - f_ideal_recon)')
plt.title('Time Error from Spectrum Error: Real Part')
plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_coords.astype(float), delta_f_imag_due_to_F_err.astype(float), label='Imag Part (f_total_recon - f_ideal_recon)')
plt.title('Time Error from Spectrum Error: Imag Part')
plt.xlabel('Time (t)')
plt.grid(True)
plt.legend()

plt.suptitle(f'Time Domain Error due to Imperfect Spectrum (ICFT L={L_param_icft_optimized}, M={M_param_icft_optimized})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("error_viz_delta_f_due_to_F_err.png")
plt.close()
print("Saved plot for delta_f_due_to_F_err(t) to error_viz_delta_f_due_to_F_err.png")

# 6.3 ICFT Intrinsic Error delta_f_intrinsic(t)
delta_f_intrinsic_real = np.real(f_reconstructed_ideal_spec) - np.real(f_original)
delta_f_intrinsic_imag = np.imag(f_reconstructed_ideal_spec) - np.imag(f_original) # f_original is real

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(t_coords.astype(float), delta_f_intrinsic_real.astype(float), label='Real Part (f_ideal_recon - f_original)')
plt.title('ICFT Intrinsic Error: Real Part')
plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_coords.astype(float), delta_f_intrinsic_imag.astype(float), label='Imag Part (f_ideal_recon - f_original)')
plt.title('ICFT Intrinsic Error: Imaginary Part')
plt.xlabel('Time (t)')
plt.grid(True)
plt.legend()

plt.suptitle(f'Detailed ICFT Intrinsic Error (ICFT L={L_param_icft_optimized}, M={M_param_icft_optimized})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("error_viz_delta_f_intrinsic.png")
plt.close()
print("Saved plot for delta_f_intrinsic(t) to error_viz_delta_f_intrinsic.png")

print("\nAll experiments and plotting complete.") 