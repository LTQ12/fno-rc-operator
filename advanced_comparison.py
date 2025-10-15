import numpy as np
import scipy.fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import time
import scipy.integrate
from scipy import signal

# Import the custom Fourier operations
import fourier_ops_custom

# --- Matplotlib Configuration ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# Create directory for plots
plots_dir = "advanced_comparison_plots"
os.makedirs(plots_dir, exist_ok=True)

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
        try:
            real_part_ld, _ = scipy.integrate.quad(_integrand_real_direct_nested, t_min, t_max, 
                                                  args=(u_val_float, sig_func), 
                                                  epsabs=q_epsabs, epsrel=q_epsrel, 
                                                  limit=q_limit, points=[t_min, t_max])
            imag_part_ld, _ = scipy.integrate.quad(_integrand_imag_direct_nested, t_min, t_max, 
                                                  args=(u_val_float, sig_func), 
                                                  epsabs=q_epsabs, epsrel=q_epsrel, 
                                                  limit=q_limit, points=[t_min, t_max])
            return np.complex128(np.longdouble(real_part_ld) + 1j * np.longdouble(imag_part_ld))
        except Exception as e:
            if verbose_level > 0:
                print(f"Warning: Integration failed for frequency {u_val_float}Hz: {str(e)}")
            return 0.0 + 0.0j

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

def experiment_1_high_dynamic_range():
    """
    Experiment 1: High Dynamic Range Signal Analysis
    Compares CFT and FFT in detecting weak signals in the presence of strong signals
    """
    print("\n=== Experiment 1: High Dynamic Range Signal Analysis ===")
    
    # Signal parameters
    fs = 100  # Sampling rate
    t_min, t_max = 0, 2
    t = np.linspace(t_min, t_max, int((t_max - t_min) * fs))
    
    # Create signal with high dynamic range
    def high_dynamic_signal(t):
        # Strong signal at 10Hz
        strong_signal = 1.0 * np.sin(2 * np.pi * 10 * t)
        # Weak signal at 15Hz
        weak_signal = 0.01 * np.sin(2 * np.pi * 15 * t)
        # Very weak signal at 20Hz
        very_weak_signal = 0.001 * np.sin(2 * np.pi * 20 * t)
        return strong_signal + weak_signal + very_weak_signal
    
    x = high_dynamic_signal(t)
    
    # FFT analysis
    X_fft = scipy.fft.fft(x)
    fft_freqs = scipy.fft.fftfreq(len(t), d=1/fs)
    
    # CFT analysis
    cft_spectrum = create_callable_spectrum_precomputed_quad_nodes(
        signal_func_to_integrate=high_dynamic_signal,
        t_min_integration=t_min,
        t_max_integration=t_max,
        quad_epsabs=1e-12,
        quad_epsrel=1e-12,
        quad_limit=5000,
        freq_min=-fs/2,
        freq_max=fs/2,
        L_segments=2000,
        M_cheb=35
    )
    
    X_cft = cft_spectrum(fft_freqs)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Time domain signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, 'k-', label='Original Signal')
    ax1.set_title('Time Domain Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Individual components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, 1.0 * np.sin(2 * np.pi * 10 * t), 'r-', label='Strong Signal (10Hz)')
    ax2.plot(t, 0.01 * np.sin(2 * np.pi * 15 * t), 'g-', label='Weak Signal (15Hz)')
    ax2.plot(t, 0.001 * np.sin(2 * np.pi * 20 * t), 'b-', label='Very Weak Signal (20Hz)')
    ax2.set_title('Signal Components')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Full spectrum comparison (linear scale)
    ax3 = fig.add_subplot(gs[1, 0])
    fft_freqs_shifted = np.fft.fftshift(fft_freqs)
    X_fft_shifted = np.fft.fftshift(X_fft)
    X_cft_shifted = np.fft.fftshift(X_cft)
    
    ax3.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax3.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax3.set_title('Full Spectrum (Linear Scale)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim(0, 25)
    ax3.grid(True)
    ax3.legend()
    
    # 4. Full spectrum comparison (log scale)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax4.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax4.set_title('Full Spectrum (Log Scale)')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude (log)')
    ax4.set_xlim(0, 25)
    ax4.set_yscale('log')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Zoomed spectrum around weak signals
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax5.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax5.set_title('Zoomed Spectrum (14-21 Hz)')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude')
    ax5.set_xlim(14, 21)
    ax5.set_yscale('log')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Phase comparison
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(fft_freqs_shifted, np.angle(X_fft_shifted), 'b-', label='FFT Phase', alpha=0.7)
    ax6.plot(fft_freqs_shifted, np.angle(X_cft_shifted), 'r-', label='CFT Phase', linewidth=2)
    ax6.set_title('Phase Spectrum')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Phase (rad)')
    ax6.set_xlim(0, 25)
    ax6.grid(True)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'experiment1_high_dynamic_range.png'))
    plt.close()

def experiment_2_non_stationary():
    """
    Experiment 2: Non-Stationary Signal Analysis
    Compares CFT and FFT in analyzing signals with time-varying frequency
    """
    print("\n=== Experiment 2: Non-Stationary Signal Analysis ===")
    
    # Signal parameters
    fs = 100  # Sampling rate
    t_min, t_max = 0, 2
    t = np.linspace(t_min, t_max, int((t_max - t_min) * fs))
    
    # Create linear chirp signal
    def chirp_signal(t):
        # Frequency varies from 5Hz to 20Hz
        f0, f1 = 5, 20
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * (t_max - t_min)))
        return np.sin(phase)
    
    x = chirp_signal(t)
    
    # FFT analysis
    X_fft = scipy.fft.fft(x)
    fft_freqs = scipy.fft.fftfreq(len(t), d=1/fs)
    
    # CFT analysis
    cft_spectrum = create_callable_spectrum_precomputed_quad_nodes(
        signal_func_to_integrate=chirp_signal,
        t_min_integration=t_min,
        t_max_integration=t_max,
        quad_epsabs=1e-12,
        quad_epsrel=1e-12,
        quad_limit=5000,
        freq_min=-fs/2,
        freq_max=fs/2,
        L_segments=2000,
        M_cheb=35
    )
    
    X_cft = cft_spectrum(fft_freqs)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Time domain signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, 'k-', label='Chirp Signal')
    ax1.set_title('Time Domain Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Instantaneous frequency
    ax2 = fig.add_subplot(gs[0, 1])
    f0, f1 = 5, 20
    inst_freq = f0 + (f1 - f0) * t / (t_max - t_min)
    ax2.plot(t, inst_freq, 'r-', label='Instantaneous Frequency')
    ax2.set_title('Instantaneous Frequency')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Full spectrum comparison
    ax3 = fig.add_subplot(gs[1, 0])
    fft_freqs_shifted = np.fft.fftshift(fft_freqs)
    X_fft_shifted = np.fft.fftshift(X_fft)
    X_cft_shifted = np.fft.fftshift(X_cft)
    
    ax3.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax3.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax3.set_title('Full Spectrum')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim(0, 25)
    ax3.grid(True)
    ax3.legend()
    
    # 4. Zoomed spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax4.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax4.set_title('Zoomed Spectrum (4-21 Hz)')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim(4, 21)
    ax4.grid(True)
    ax4.legend()
    
    # 5. Phase spectrum
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(fft_freqs_shifted, np.angle(X_fft_shifted), 'b-', label='FFT Phase', alpha=0.7)
    ax5.plot(fft_freqs_shifted, np.angle(X_cft_shifted), 'r-', label='CFT Phase', linewidth=2)
    ax5.set_title('Phase Spectrum')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Phase (rad)')
    ax5.set_xlim(0, 25)
    ax5.grid(True)
    ax5.legend()
    
    # 6. Time-frequency representation
    ax6 = fig.add_subplot(gs[2, 1])
    # Compute spectrogram
    f, t_spec, Sxx = signal.spectrogram(x, fs, nperseg=64, noverlap=32)
    ax6.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    ax6.set_title('Spectrogram')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Frequency (Hz)')
    ax6.set_ylim(0, 25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'experiment2_non_stationary.png'))
    plt.close()

def experiment_3_multi_scale():
    """
    Experiment 3: Multi-Scale Signal Analysis
    Compares CFT and FFT in analyzing signals with multiple time scales
    """
    print("\n=== Experiment 3: Multi-Scale Signal Analysis ===")
    
    # Signal parameters
    fs = 100  # Sampling rate
    t_min, t_max = 0, 2
    t = np.linspace(t_min, t_max, int((t_max - t_min) * fs))
    
    # Create multi-scale signal
    def multi_scale_signal(t):
        # Fast oscillation
        fast_osc = np.sin(2 * np.pi * 20 * t)
        # Slow envelope
        slow_env = np.exp(-(t - 1)**2 / 0.5)
        return fast_osc * slow_env
    
    x = multi_scale_signal(t)
    
    # FFT analysis
    X_fft = scipy.fft.fft(x)
    fft_freqs = scipy.fft.fftfreq(len(t), d=1/fs)
    
    # CFT analysis
    cft_spectrum = create_callable_spectrum_precomputed_quad_nodes(
        signal_func_to_integrate=multi_scale_signal,
        t_min_integration=t_min,
        t_max_integration=t_max,
        quad_epsabs=1e-12,
        quad_epsrel=1e-12,
        quad_limit=5000,
        freq_min=-fs/2,
        freq_max=fs/2,
        L_segments=2000,
        M_cheb=35
    )
    
    X_cft = cft_spectrum(fft_freqs)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Time domain signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, 'k-', label='Multi-Scale Signal')
    ax1.set_title('Time Domain Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Signal components
    ax2 = fig.add_subplot(gs[0, 1])
    fast_osc = np.sin(2 * np.pi * 20 * t)
    slow_env = np.exp(-(t - 1)**2 / 0.5)
    ax2.plot(t, fast_osc, 'r-', label='Fast Oscillation', alpha=0.5)
    ax2.plot(t, slow_env, 'g-', label='Slow Envelope')
    ax2.set_title('Signal Components')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Full spectrum comparison
    ax3 = fig.add_subplot(gs[1, 0])
    fft_freqs_shifted = np.fft.fftshift(fft_freqs)
    X_fft_shifted = np.fft.fftshift(X_fft)
    X_cft_shifted = np.fft.fftshift(X_cft)
    
    ax3.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax3.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax3.set_title('Full Spectrum')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim(0, 30)
    ax3.grid(True)
    ax3.legend()
    
    # 4. Zoomed spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax4.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax4.set_title('Zoomed Spectrum (18-22 Hz)')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim(18, 22)
    ax4.grid(True)
    ax4.legend()
    
    # 5. Phase spectrum
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(fft_freqs_shifted, np.angle(X_fft_shifted), 'b-', label='FFT Phase', alpha=0.7)
    ax5.plot(fft_freqs_shifted, np.angle(X_cft_shifted), 'r-', label='CFT Phase', linewidth=2)
    ax5.set_title('Phase Spectrum')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Phase (rad)')
    ax5.set_xlim(0, 30)
    ax5.grid(True)
    ax5.legend()
    
    # 6. Time-frequency representation
    ax6 = fig.add_subplot(gs[2, 1])
    # Compute spectrogram
    f, t_spec, Sxx = signal.spectrogram(x, fs, nperseg=64, noverlap=32)
    ax6.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    ax6.set_title('Spectrogram')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Frequency (Hz)')
    ax6.set_ylim(0, 30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'experiment3_multi_scale.png'))
    plt.close()

def experiment_5_boundary_effects():
    """
    Experiment 5: Boundary Effects Analysis
    Compares CFT and FFT in handling signals with boundary discontinuities
    """
    print("\n=== Experiment 5: Boundary Effects Analysis ===")
    
    # Signal parameters
    fs = 100  # Sampling rate
    t_min, t_max = 0, 2
    t = np.linspace(t_min, t_max, int((t_max - t_min) * fs))
    
    # Create signal with boundary discontinuities
    def boundary_signal(t):
        # Base signal
        base = np.sin(2 * np.pi * 10 * t)
        # Add discontinuities at boundaries
        return base + 0.5 * (t < 0.1) - 0.5 * (t > 1.9)
    
    x = boundary_signal(t)
    
    # FFT analysis
    X_fft = scipy.fft.fft(x)
    fft_freqs = scipy.fft.fftfreq(len(t), d=1/fs)
    
    # CFT analysis
    cft_spectrum = create_callable_spectrum_precomputed_quad_nodes(
        signal_func_to_integrate=boundary_signal,
        t_min_integration=t_min,
        t_max_integration=t_max,
        quad_epsabs=1e-12,
        quad_epsrel=1e-12,
        quad_limit=5000,
        freq_min=-fs/2,
        freq_max=fs/2,
        L_segments=2000,
        M_cheb=35
    )
    
    X_cft = cft_spectrum(fft_freqs)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Full time domain signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, 'k-', label='Signal with Boundary Effects')
    ax1.set_title('Full Time Domain Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Zoomed time domain at boundaries
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, x, 'k-', label='Signal')
    ax2.set_title('Zoomed View at Boundaries')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(0, 0.2)  # Zoom in on first boundary
    ax2.grid(True)
    ax2.legend()
    
    # 3. Full spectrum comparison
    ax3 = fig.add_subplot(gs[1, 0])
    fft_freqs_shifted = np.fft.fftshift(fft_freqs)
    X_fft_shifted = np.fft.fftshift(X_fft)
    X_cft_shifted = np.fft.fftshift(X_cft)
    
    ax3.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax3.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax3.set_title('Full Spectrum')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim(0, 20)
    ax3.grid(True)
    ax3.legend()
    
    # 4. Zoomed spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(fft_freqs_shifted, np.abs(X_fft_shifted), 'b-', label='FFT', alpha=0.7)
    ax4.plot(fft_freqs_shifted, np.abs(X_cft_shifted), 'r-', label='CFT', linewidth=2)
    ax4.set_title('Zoomed Spectrum (0-5 Hz)')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim(0, 5)
    ax4.grid(True)
    ax4.legend()
    
    # 5. Phase spectrum
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(fft_freqs_shifted, np.angle(X_fft_shifted), 'b-', label='FFT Phase', alpha=0.7)
    ax5.plot(fft_freqs_shifted, np.angle(X_cft_shifted), 'r-', label='CFT Phase', linewidth=2)
    ax5.set_title('Phase Spectrum')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Phase (rad)')
    ax5.set_xlim(0, 20)
    ax5.grid(True)
    ax5.legend()
    
    # 6. Time-frequency representation
    ax6 = fig.add_subplot(gs[2, 1])
    # Compute spectrogram
    f, t_spec, Sxx = signal.spectrogram(x, fs, nperseg=64, noverlap=32)
    ax6.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    ax6.set_title('Spectrogram')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Frequency (Hz)')
    ax6.set_ylim(0, 20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'experiment5_boundary_effects.png'))
    plt.close()

def experiment_6_computation_efficiency():
    """
    Experiment 6: Computation Efficiency Analysis
    Compares computation time and accuracy of CFT and FFT for different signal lengths
    """
    print("\n=== Experiment 6: Computation Efficiency Analysis ===")
    
    # Test different signal lengths
    lengths = [100, 500, 1000, 2000, 5000]
    fft_times = []
    cft_times = []
    fft_errors = []
    cft_errors = []
    
    # Reference signal parameters
    fs = 100  # Sampling rate
    t_min, t_max = 0, 2
    
    # Test signal function
    def test_signal(t):
        return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    
    for length in lengths:
        print(f"Testing signal length: {length}")
        t = np.linspace(t_min, t_max, length)
        x = test_signal(t)
        
        # FFT timing
        start_time = time.time()
        X_fft = scipy.fft.fft(x)
        fft_time = time.time() - start_time
        fft_times.append(fft_time)
        
        # CFT timing
        start_time = time.time()
        cft_spectrum = create_callable_spectrum_precomputed_quad_nodes(
            signal_func_to_integrate=test_signal,
            t_min_integration=t_min,
            t_max_integration=t_max,
            quad_epsabs=1e-12,
            quad_epsrel=1e-12,
            quad_limit=5000,
            freq_min=-fs/2,
            freq_max=fs/2,
            L_segments=2000,
            M_cheb=35
        )
        X_cft = cft_spectrum(scipy.fft.fftfreq(length, d=1/fs))
        cft_time = time.time() - start_time
        cft_times.append(cft_time)
        
        # Calculate errors (using clean signal as reference)
        fft_error = np.mean(np.abs(X_fft - X_cft))
        fft_errors.append(fft_error)
        cft_errors.append(0)  # CFT is our reference
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Computation time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(lengths, fft_times, 'b-o', label='FFT')
    ax1.plot(lengths, cft_times, 'r-o', label='CFT')
    ax1.set_title('Computation Time vs Signal Length')
    ax1.set_xlabel('Signal Length')
    ax1.set_ylabel('Time (s)')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Error comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(lengths, fft_errors, 'b-o', label='FFT Error')
    ax2.set_title('FFT Error vs Signal Length')
    ax2.set_xlabel('Signal Length')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_yscale('log')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Speedup ratio
    ax3 = fig.add_subplot(gs[1, 0])
    speedup = np.array(cft_times) / np.array(fft_times)
    ax3.plot(lengths, speedup, 'g-o')
    ax3.set_title('CFT/FFT Speedup Ratio')
    ax3.set_xlabel('Signal Length')
    ax3.set_ylabel('Speedup Ratio')
    ax3.grid(True)
    
    # 4. Memory usage estimation
    ax4 = fig.add_subplot(gs[1, 1])
    fft_memory = np.array(lengths) * 8  # 8 bytes per complex number
    cft_memory = np.array(lengths) * 8 * 2  # Rough estimate for CFT
    ax4.plot(lengths, fft_memory, 'b-o', label='FFT Memory')
    ax4.plot(lengths, cft_memory, 'r-o', label='CFT Memory')
    ax4.set_title('Estimated Memory Usage')
    ax4.set_xlabel('Signal Length')
    ax4.set_ylabel('Memory (bytes)')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Error vs Time tradeoff
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(fft_times, fft_errors, c='b', label='FFT')
    ax5.scatter(cft_times, cft_errors, c='r', label='CFT')
    ax5.set_title('Error vs Computation Time')
    ax5.set_xlabel('Computation Time (s)')
    ax5.set_ylabel('Mean Absolute Error')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Efficiency score
    ax6 = fig.add_subplot(gs[2, 1])
    efficiency_score = 1 / (np.array(fft_errors) * np.array(fft_times))
    ax6.plot(lengths, efficiency_score, 'b-o')
    ax6.set_title('FFT Efficiency Score (1/(error*time))')
    ax6.set_xlabel('Signal Length')
    ax6.set_ylabel('Efficiency Score')
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'experiment6_computation_efficiency.png'))
    plt.close()

if __name__ == "__main__":
    print("="*40)
    print("      Advanced CFT vs. FFT Comparison Experiments")
    print("="*40)
    
    # Run all experiments except experiment 4
    experiment_1_high_dynamic_range()
    experiment_2_non_stationary()
    experiment_3_multi_scale()
    experiment_5_boundary_effects()
    experiment_6_computation_efficiency()
    
    print("\nAll experiments completed. Results saved to '{}' directory.".format(plots_dir)) 