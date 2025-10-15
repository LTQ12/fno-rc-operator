import numpy as np
from scipy.special import factorial
import time

# Note: Using np.longdouble for intermediate calculations where possible,
# and np.clongdouble for complex intermediate calculations.

def monomial_fourier_transform(m, u_scaled):
    """
    Calculates the Fourier transform of a monomial t^m over [-1, 1].
    Uses Taylor series approximation for small |W| = |2*pi*u_scaled|.

    Args:
        m (int): The exponent of the monomial.
        u_scaled (float/longdouble): The scaled frequency (u * a).

    Returns:
        complex: The Fourier transform of t^m over [-1, 1].
    """
    # Handle u_scaled = 0 case first
    if np.isclose(u_scaled, 0):
        if m == 0: return np.clongdouble(2.0)
        elif m % 2 != 0: return np.clongdouble(0.0)
        else: return np.clongdouble(2.0 / (m + 1))

    # Calculate W = j * 2 * pi * u_scaled using high precision
    W = np.clongdouble(2j * np.pi * u_scaled)
    abs_W = np.abs(W)

    # Define Taylor series threshold and truncation
    taylor_threshold = 0.1 # Adjust based on testing
    S = 30 # Number of terms for Taylor series

    # Use Taylor series for small |W|
    if abs_W < taylor_threshold:
        taylor_sum = np.clongdouble(0.0)
        W_pow_s = np.clongdouble(1.0) # W^0
        fact_s = np.longdouble(1.0)    # 0!
        term_m_pow_neg1 = np.longdouble(-1.0)**m

        for s in range(S + 1):
            # Calculate term: [(-1)^s + (-1)^m] / (m + s + 1)
            term_s_pow_neg1 = np.longdouble(-1.0)**s
            term_numerator = term_s_pow_neg1 + term_m_pow_neg1
            term_denominator = np.longdouble(m + s + 1)

            if np.isclose(term_denominator, 0):
                term_frac = np.longdouble(0.0)
            else:
                term_frac = term_numerator / term_denominator

            # Calculate W^s / s! term carefully
            taylor_component = W_pow_s / fact_s
            
            # Check for NaN/Inf before adding
            if np.isnan(taylor_component) or np.isinf(taylor_component):
                 # print(f"Warning: Taylor component NaN/Inf at s={s}. Stopping series.")
                 break

            taylor_sum += term_frac * taylor_component

            # Update W^s and s! for next iteration
            if s < S:
                W_pow_s *= W
                # Check factorial calculation safety
                next_fact_s_val = s + 1
                if next_fact_s_val > 170: # Avoid factorial overflow for float64 based fact_s
                     # print(f"Warning: Factorial s+1={next_fact_s_val} likely to overflow. Stopping series.")
                     break
                fact_s *= np.longdouble(next_fact_s_val)
                if np.isinf(fact_s) or np.isclose(fact_s, 0):
                    # print(f"Warning: Factorial calculation issue at s={s+1}. Stopping series.")
                    break
        return taylor_sum

    # Use original formula for larger |W|
    else:
        term1_sum = np.clongdouble(0.0)
        term2_sum = np.clongdouble(0.0)
        
        try: # Protect factorial calculations
            fact_m = np.longdouble(factorial(m, exact=False))
            if np.isinf(fact_m): raise OverflowError("m! overflow")
            
            w_inv = np.clongdouble(1.0) / W
            w_power_s_plus_1 = w_inv # W^-(s+1) for s=0

            for s in range(m + 1):
                m_minus_s = m - s
                fact_m_minus_s = np.longdouble(factorial(m_minus_s, exact=False))
                if np.isinf(fact_m_minus_s): raise OverflowError("(m-s)! overflow")
                if np.isclose(fact_m_minus_s, 0): # Factorial of negative is undefined (or gamma pole), treat as error/skip
                    coeff = np.clongdouble(0.0)
                else:
                    coeff = fact_m / fact_m_minus_s * w_power_s_plus_1

                if m_minus_s % 2 == 0: term1_sum += coeff
                else: term1_sum -= coeff
                term2_sum += coeff

                if s < m:
                    w_power_s_plus_1 *= w_inv
                    
        except (OverflowError, ValueError) as e:
            # print(f"Warning: Factorial issue in original formula (m={m}, s={s}): {e}. Returning 0.")
            return np.clongdouble(0.0) # Return 0 on factorial error


        exp_W = np.exp(W)
        exp_neg_W = np.exp(-W)
        
        # Final calculation, check for potential NaN/Inf from large exponents
        result = exp_W * term1_sum - exp_neg_W * term2_sum
        if np.isnan(result) or np.isinf(result):
             # print(f"Warning: Result NaN/Inf in original formula (m={m}, W={W}). Returning 0.")
             return np.clongdouble(0.0)
             
        return result


def cft_element(f, xl, xl1, u, M):
    """
    Calculates the CFT for a function f over a single element [xl, xl1] using Dkm method.
    """
    delta = xl1 - xl
    a = delta / 2.0
    hl = (xl1 + xl) / 2.0 # Midpoint

    if np.isclose(a, 0):
        return np.clongdouble(0.0)

    # Standard interval [-1, 1] interpolation points
    t_points = np.linspace(-1.0, 1.0, M + 1, dtype=np.longdouble)
    # Physical interval [xl, xl1] interpolation points
    x_points = a * t_points + hl
    
    # Get function values at physical points
    try:
        # Ensure f_values are calculated with high precision if possible
        f_values = np.array([f(x) for x in x_points], dtype=np.clongdouble)
    except Exception as e:
        print(f"Error evaluating function f at points {x_points} in interval [{xl}, {xl1}]: {e}")
        return np.clongdouble(np.nan) # Indicate error

    # --- Calculate coefficients D_{k,m} ---
    # D[k, m] is the coefficient of t^m in the k-th Lagrange basis polynomial L_k(t)
    # Note: np.poly1d might use float64 internally, which could be a precision bottleneck.
    D = np.zeros((M + 1, M + 1), dtype=np.clongdouble)
    for k in range(M + 1):
        numerator_poly = np.poly1d([1.0]) # Start with polynomial '1'
        denominator_val = np.longdouble(1.0)
        tk = t_points[k]

        for i in range(M + 1):
            if i != k:
                ti = t_points[i]
                # Multiply numerator by (t - t_i)
                numerator_poly = np.polymul(numerator_poly, np.poly1d([1.0, -ti]))
                # Multiply denominator by (t_k - t_i)
                diff = tk - ti
                if np.isclose(diff, 0): # Should not happen with linspace points unless M=0?
                    print(f"Warning: Denominator close to zero in Dkm calculation (k={k}, i={i}).")
                    denominator_val = np.longdouble(1e-100) # Avoid division by zero
                else:
                    denominator_val *= diff

        if np.isclose(denominator_val, 0):
             coeffs_Lk = np.zeros(M + 1) # Avoid division by zero
        else:
             # Get coefficients of numerator polynomial (highest power first)
             coeffs_numerator = numerator_poly.coeffs
             # Divide by denominator to get coefficients of L_k(t)
             coeffs_Lk = coeffs_numerator / denominator_val

        # Ensure coeffs_Lk has length M+1 (pad if needed)
        if len(coeffs_Lk) < M + 1:
            coeffs_Lk = np.pad(coeffs_Lk, (M + 1 - len(coeffs_Lk), 0), 'constant')
        elif len(coeffs_Lk) > M + 1:
             coeffs_Lk = coeffs_Lk[-(M+1):] # Should not happen, but safeguard

        # Store coefficients D[k, m] (m-th power coefficient)
        # np.poly1d coeffs are highest power first, D[k,m] needs m-th power coeff.
        # So, reverse the order.
        D[k, :] = coeffs_Lk[::-1]


    # --- Calculate Integral ---
    integral_sum = np.clongdouble(0.0)
    u_scaled = np.longdouble(u * a) # Scaled frequency for monomial_fourier_transform

    for k in range(M + 1):
        term_k = np.clongdouble(0.0)
        for m in range(M + 1):
            ft_monomial = monomial_fourier_transform(m, u_scaled)
            term_k += np.clongdouble(D[k, m]) * ft_monomial
        
        integral_sum += np.clongdouble(f_values[k]) * term_k


    # Final result: a * exp(-j2pi*u*hl) * IntegralSum
    phase_factor = np.exp(np.clongdouble(-2j * np.pi * u * hl))
    result_element = np.clongdouble(a) * phase_factor * integral_sum
    
    # Check for NaN/Inf in the final result
    if np.isnan(result_element) or np.isinf(result_element):
        # print(f"Warning: Result NaN/Inf in cft_element (u={u}, interval=[{xl},{xl1}]). Returning 0.")
        return np.clongdouble(0.0)

    return result_element


def cft_dkm(f, p0, p1, u_array, L: int, M: int) -> np.ndarray:
    """
    Calculates the Conformal Fourier Transform (CFT) of f over [p0, p1]
    for multiple frequency values using the Dkm coefficient method.

    Args:
        f (callable): The function to transform.
        p0 (float): The start of the interval.
        p1 (float): The end of the interval.
        u_array (np.ndarray): Array of frequencies to compute the transform for.
        L (int): The number of elements (subintervals).
        M (int): The interpolation order.

    Returns:
        np.ndarray: The CFT of f at the specified frequencies.
    """
    delta = (p1 - p0) / L
    K = len(u_array)
    cft_results = np.zeros(K, dtype=np.clongdouble)

    print(f"Calculating CFT using Dkm method for {K} frequencies, L={L}, M={M}...")
    start_time_cft = time.time()

    # Loop over frequencies
    for k, u in enumerate(u_array):
        element_sum = np.clongdouble(0.0)
        # Sum contributions from each element
        for l in range(L): # l from 0 to L-1
            xl = p0 + l * delta
            xl1 = p0 + (l + 1) * delta
            # Add contribution, handling potential NaN from cft_element
            contribution = cft_element(f, xl, xl1, u, M)
            if not np.isnan(contribution):
                 element_sum += contribution
            # else: # Optional: Log skipped elements due to errors
                 # print(f"Skipping element [{xl}, {xl1}] for u={u} due to NaN result.")

        cft_results[k] = element_sum

        # Progress indicator
        if K > 10 and (k + 1) % (K // 10) == 0:
             elapsed = time.time() - start_time_cft
             print(f"  ... processed {k+1}/{K} frequencies ({elapsed:.2f}s)")

    end_time_cft = time.time()
    print(f"CFT (Dkm method) calculation time: {end_time_cft - start_time_cft:.4f}s")
    return cft_results


def icft_dkm(spectrum_callable, freq_min, freq_max, t_values: np.ndarray, L: int, M: int) -> np.ndarray:
    """
    Calculates the Inverse CFT using the Dkm CFT method structure.
    Treats spectrum as function F(u), interval [freq_min, freq_max], target 'frequencies' are -t.

    Args:
        spectrum_callable (callable): A function F(u) representing the spectrum.
        freq_min (float): Minimum frequency of the spectrum interval.
        freq_max (float): Maximum frequency of the spectrum interval.
        t_values (np.ndarray): Array of time points to compute the inverse transform for.
        L (int): Number of elements in frequency domain.
        M (int): Interpolation order in frequency domain.

    Returns:
        np.ndarray: The reconstructed signal f(t).
    """
    print("Calculating Inverse CFT using Dkm method...")
    # Target 'frequencies' for the inverse transform are -t
    # For ICFT formula: Integral F(u) * exp(j * 2 * pi * u * t) du
    # Let u' = -t. The kernel is exp(-j * 2 * pi * u * (-u'))
    # So we call CFT with frequency u' = -t
    target_freqs = -t_values

    # Call the forward CFT routine with spectrum as function and -t as frequencies
    # The 'interval' is the frequency range [freq_min, freq_max]
    icft_raw = cft_dkm(spectrum_callable, freq_min, freq_max, target_freqs, L, M)

    # Scaling: Inverse FT definition usually involves 1/(2pi) or similar,
    # but the discrete approximation requires scaling by du.
    # Estimate du assuming uniform sampling in the original spectrum.
    # This needs careful handling depending on how the original spectrum was sampled.
    num_t_points = len(t_values) # Use this as a proxy for N_freqs? Risky.
    du = 1.0 # Default scaling factor

    if num_t_points > 1:
        # Use provided freq range to estimate du
        freq_range = freq_max - freq_min
        # Estimate number of points for du calculation using len(t_values) as proxy.
        N_freqs_est = num_t_points
        if N_freqs_est > 1 and not np.isclose(freq_range, 0):
             # This assumes t_values somehow correspond to the original frequency grid size
             du = freq_range / (N_freqs_est -1) if N_freqs_est > 1 else freq_range # Or N_freqs_est? Need consistency
             print(f"Applying ICFT scaling with estimated du = {du:.4e}")
        else:
             print("Warning: Could not reliably estimate du from freq range and t_values length. Using du=1.0.")
             du = 1.0
    else:
        print("Warning: Cannot estimate du for ICFT scaling (len(t_values) <= 1). Using du=1.0.")
        du = 1.0

    # Apply scaling
    scaled_icft = icft_raw * du

    print("Inverse CFT (Dkm method) calculation done.")
    return scaled_icft 