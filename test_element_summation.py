import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import chebpts2
from cft_direct import monomial_fourier_transform # Assuming cft_direct.py is in the PYTHONPATH or same directory
from scipy.integrate import quad # Added for benchmark integration

# --- Test Parameters ---
P0_global = -1.0  # Global interval start
P1_global = 1.0  # Global interval end
L_divs = 10      # Number of subintervals (elements)
M_order = 10     # Polynomial interpolation order for each element

# --- Test Function (e.g., Gaussian) ---
def gaussian_func(x, mu=0, sigma=0.05):
    """Simple Gaussian function for testing.""" 
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

func_to_analyze = gaussian_func
func_name = 'Gaussian_sigma0.05'

# --- Select a specific subinterval to analyze (0 to L_divs-1) ---
l_test_idx = L_divs // 2  # Analyze the middle subinterval

# --- Frequencies to test ---
# Pick some representative u values. These would typically come from fftfreq.
# For now, let's use a few illustrative values.
u_test_values = [0.1, 1.0, 5.0, 10.0, 50.0] 
# u_test_values = [50.0] # For focused test

def main():
    print(f"--- Testing Element Summation: FT(Polynomial) vs FT(True Sub-Function) ---")
    print(f"Function: {func_name}, Global Interval: [{P0_global}, {P1_global}], L={L_divs}, M={M_order}")
    print(f"Analyzing subinterval index l = {l_test_idx}")

    # 1. Calculate parameters for the chosen subinterval
    delta_global = (P1_global - P0_global) / L_divs
    xl = P0_global + l_test_idx * delta_global
    xl1 = P0_global + (l_test_idx + 1) * delta_global
    a_l = (xl1 - xl) / 2.0
    hl_l = (xl1 + xl) / 2.0
    print(f"Subinterval [{xl:.4f}, {xl1:.4f}]; a_l = {a_l:.4f}; hl_l = {hl_l:.4f}")

    # 2. Get Chebyshev interpolation points and function values
    if M_order == 0:
        t_points_std = np.array([0.0], dtype=np.float64)
    else:
        t_points_std = chebpts2(M_order + 1).astype(np.float64)
    
    f_values_at_cheb_nodes = np.array([func_to_analyze(a_l * t_s + hl_l) for t_s in t_points_std])

    # 3. Get B_coeffs from polynomial fit (real part only for Gaussian)
    # Ensure B_coeffs has length M_order + 1 and is complex for consistency with cft_element
    B_coeffs_poly_obj_callable = None # To store the Polynomial object
    if np.iscomplexobj(f_values_at_cheb_nodes) or np.any(np.iscomplex(f_values_at_cheb_nodes)):
        print("Processing complex function values for B_coeffs")
        f_real = np.real(f_values_at_cheb_nodes).astype(np.float64)
        f_imag = np.imag(f_values_at_cheb_nodes).astype(np.float64)
        poly_real_obj = Polynomial.fit(t_points_std, f_real, M_order)
        poly_imag_obj = Polynomial.fit(t_points_std, f_imag, M_order)
        B_coeffs_real = poly_real_obj.convert().coef
        B_coeffs_imag = poly_imag_obj.convert().coef
        
        # Pad if necessary
        if len(B_coeffs_real) < M_order + 1: B_coeffs_real = np.pad(B_coeffs_real, (0, M_order + 1 - len(B_coeffs_real)))
        if len(B_coeffs_imag) < M_order + 1: B_coeffs_imag = np.pad(B_coeffs_imag, (0, M_order + 1 - len(B_coeffs_imag)))
        B_coeffs = B_coeffs_real[:M_order+1] + 1j * B_coeffs_imag[:M_order+1] # Ensure correct length
        # For benchmark, we need a callable that evaluates the complex polynomial
        def complex_poly_eval(t_val):
            return poly_real_obj(t_val) + 1j * poly_imag_obj(t_val)
        B_coeffs_poly_obj_callable = complex_poly_eval
    else:
        print("Processing real function values for B_coeffs")
        f_real_data = np.real(f_values_at_cheb_nodes).astype(np.float64)
        poly_fit_obj = Polynomial.fit(t_points_std, f_real_data, M_order)
        B_coeffs_poly_obj_callable = poly_fit_obj # This is already a callable Polynomial object
        B_coeffs_real = poly_fit_obj.convert().coef
        if len(B_coeffs_real) < M_order + 1: B_coeffs_real = np.pad(B_coeffs_real, (0, M_order + 1 - len(B_coeffs_real)))
        B_coeffs = B_coeffs_real[:M_order+1].astype(np.complex128) # Ensure correct length and complex type
    
    print(f"B_coeffs (shape {B_coeffs.shape}, type {B_coeffs.dtype}):")
    for m_idx, b_val in enumerate(B_coeffs):
        print(f"  B_{m_idx:02d}: {b_val.real:+.6e} {b_val.imag:+.6e}j")

    # 4. Loop through test frequencies
    for u_val in u_test_values:
        print(f"\n------ Testing for u = {u_val:.4f} ------")
        w_s = np.longdouble(u_val * a_l)
        print(f"  w_s (u * a_l for monomial_ft) = {w_s:.6e}")

        # 5. Calculate F_m_values
        F_m_values = np.array([monomial_fourier_transform(m, w_s) for m in range(M_order + 1)], dtype=np.clongdouble)
        print(f"  F_m_values (type {F_m_values.dtype}):")
        # Limiting printout for F_m if too long, or select specific m's
        # for m_idx, f_val in enumerate(F_m_values):
        #     print(f"    F_{m_idx:02d}: {f_val.real:+.6e} {f_val.imag:+.6e}j")

        # 6. Calculate and print each term B_m * F_m and their sum
        print(f"  Terms B_m * F_m (B_coeffs type: {B_coeffs.dtype}, F_m_values type: {F_m_values.dtype}):")
        calculated_sum_direct_method = np.clongdouble(0.0)
        terms_array = np.empty_like(F_m_values, dtype=np.clongdouble)

        for m_idx in range(M_order + 1):
            # Explicitly cast B_coeffs[m_idx] to clongdouble for the multiplication if it's not already
            # B_m is complex128, F_m is clongdouble. NumPy handles mixed type by upcasting.
            term_m = np.clongdouble(B_coeffs[m_idx]) * F_m_values[m_idx]
            terms_array[m_idx] = term_m
            calculated_sum_direct_method += term_m
            # Only print first few and last few terms if M_order is large to avoid excessive output
            if M_order <= 10 or m_idx < 3 or m_idx > M_order - 3:
                 print(f"    Term {m_idx:02d} (B_{m_idx:02d}*F_{m_idx:02d}): {term_m.real:+.6e} {term_m.imag:+.6e}j,  |Term|: {np.abs(term_m):.6e}")
            elif m_idx == 3:
                 print(f"      ... (terms {m_idx} to {M_order-3} omitted for brevity) ...")
        
        print(f"  FT{{P(t')}} [Sum(B_m*FT(t^m))] = {calculated_sum_direct_method.real:+.6e} {calculated_sum_direct_method.imag:+.6e}j")

        # Benchmark 1: Integral of P(t') * exp(-j*2*pi*w_s*t') dt' (should be close to above sum)
        def poly_integrand(t_prime, w_scaled_val):
            return B_coeffs_poly_obj_callable(float(t_prime)) * np.exp(-2j * np.pi * np.longdouble(w_scaled_val) * float(t_prime))
        integral_poly_real, _ = quad(lambda tp: np.real(poly_integrand(tp, w_s)), -1.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)
        integral_poly_imag, _ = quad(lambda tp: np.imag(poly_integrand(tp, w_s)), -1.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)
        benchmark_ft_poly_quad = np.clongdouble(integral_poly_real + 1j * integral_poly_imag)
        print(f"  FT{{P(t')}} [Quad Benchmark]    = {benchmark_ft_poly_quad.real:+.6e} {benchmark_ft_poly_quad.imag:+.6e}j")
        error_sum_vs_poly_quad = np.abs(calculated_sum_direct_method - benchmark_ft_poly_quad)
        print(f"    Abs Error (Sum Method vs Poly Quad) = {error_sum_vs_poly_quad:.6e}")

        # Benchmark 2: Integral of g_l(t') * exp(-j*2*pi*w_s*t') dt' (FT of true sub-function)
        g_l_func = lambda t_prime_arg: func_to_analyze(a_l * t_prime_arg + hl_l)
        def true_g_integrand(t_prime, w_scaled_val):
            return g_l_func(float(t_prime)) * np.exp(-2j * np.pi * np.longdouble(w_scaled_val) * float(t_prime))
        integral_true_g_real, _ = quad(lambda tp: np.real(true_g_integrand(tp, w_s)), -1.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)
        integral_true_g_imag, _ = quad(lambda tp: np.imag(true_g_integrand(tp, w_s)), -1.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)
        benchmark_ft_true_g_quad = np.clongdouble(integral_true_g_real + 1j * integral_true_g_imag)
        print(f"  FT{{g_l(t')}} [True Sub-Func Quad] = {benchmark_ft_true_g_quad.real:+.6e} {benchmark_ft_true_g_quad.imag:+.6e}j")
        error_polyFT_vs_trueFT = np.abs(calculated_sum_direct_method - benchmark_ft_true_g_quad)
        print(f"    Abs Error (FT{{P(t')}} vs FT{{g_l(t')}}) = {error_polyFT_vs_trueFT:.6e}")

if __name__ == '__main__':
    main() 