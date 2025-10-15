import torch
from chebyshev import vectorized_batched_cft

# ==============================================================================
# Rebuilt CFT Utilities
# ==============================================================================
# This file is a reconstruction of the essential utilities needed for the
# CFT-guided G-FNO model, based on analysis of its usage in other scripts.
# ==============================================================================

def vectorized_batched_cft_decomposed(signals, L_segments, M_cheb):
    """
    Reconstruction of the 'decomposed' CFT utility.
    
    This function wraps the core `vectorized_batched_cft` function and formats
    the output to be directly consumable by the GateController network.

    Args:
        signals (torch.Tensor): The input signals, expected shape [batch, n_samples].
        L_segments (int): The number of segments for numerical integration.
        M_cheb (int): The number of Chebyshev nodes per segment.

    Returns:
        torch.Tensor: The decomposed Fourier coefficients, with shape
                      [batch, M_cheb, 2], where the last dimension holds
                      the real and imaginary parts.
    """
    if signals.dim() != 2:
        raise ValueError(f"Expected input signal to be 2D [batch, n_samples], but got shape {signals.shape}")

    n_samples = signals.shape[1]
    device = signals.device

    # The coordinate and frequency points need to be defined for the transform.
    # We assume the signal is defined over the domain [0, 1].
    t_coords = torch.linspace(0, 1, n_samples, device=device)
    
    # We only need the first M_cheb frequency modes for the gate controller.
    # We generate the frequencies that the standard FFT would produce.
    f_points = torch.fft.rfftfreq(n_samples, d=1.0/n_samples)[:M_cheb].to(device)

    # Use the core numerical integration engine to compute the CFT coefficients.
    # Note: The original `vectorized_batched_cft` expects a 3D tensor 
    # [batch, channels, n_samples], so we unsqueeze the input.
    cft_coeffs_complex = vectorized_batched_cft(
        signals=signals.unsqueeze(1), # Add a dummy channel dimension
        t_coords=t_coords,
        f_points=f_points,
        L_segments=L_segments,
        M_cheb=M_cheb
    )
    
    # The output of vectorized_batched_cft is [batch, channels, M_cheb].
    # We squeeze out the dummy channel dimension.
    cft_coeffs_complex = cft_coeffs_complex.squeeze(1) # Shape: [batch, M_cheb]

    # Decompose the complex tensor into its real and imaginary parts.
    # The torch.view_as_real function is perfect for this. It creates a
    # new last dimension of size 2.
    cft_coeffs_decomposed = torch.view_as_real(cft_coeffs_complex) # Shape: [batch, M_cheb, 2]

    return cft_coeffs_decomposed 