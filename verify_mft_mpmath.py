import numpy as np
import sympy
import mpmath as mp
import time

# Import our implementation and the SymPy version from the test script
# We need to set W_threshold_global and S_val_offset_global for our implementation
import cft_direct as cftd
W_threshold_global = 0.010  # From the scan's "best" result
S_val_offset_global = 10    # From the scan's "best" result
orig_mft = cftd.monomial_fourier_transform

def mft_numeric_test(m, u_scaled):
    # Ensure globals are set for the imported function
    cftd.W_threshold = W_threshold_global
    cftd.S_val_offset = S_val_offset_global
    return orig_mft(m, u_scaled)

# Import a potentially more stable SymPy version from the test script
def mft_symbolic_original(m, u_scaled_val):
    t, u_s_sym = sympy.symbols('t u_s_sym')
    m_int = int(m)
    integrand = (t**m_int) * sympy.exp(-sympy.I * 2 * sympy.pi * u_s_sym * t)
    try:
        integral_expr = sympy.integrate(integrand, (t, -1, 1))
    except NotImplementedError:
        print(f"Sympy could not integrate symbolically for m={m_int}.")
        return np.nan + np.nan*1j

    if np.isclose(u_scaled_val, 0):
        if m_int == 0: return 2.0 + 0.0j
        elif m_int % 2 != 0: return 0.0 + 0.0j
        else: return 2.0 / (m_int + 1) + 0.0j

    try:
        # Use mpmath's precision for SymPy evaluation as well
        result_sympy = integral_expr.subs({u_s_sym: u_scaled_val}).evalf(n=mp.dps)
        return complex(result_sympy) # Convert mpmath complex to Python complex
    except Exception as e:
        print(f"Error evaluating original sympy expression for m={m_int}, u_scaled={u_scaled_val}: {e}")
        return np.nan + np.nan*1j

def mft_mpmath_quad(m, u_scaled_val):
    """
    Calculates the Fourier Transform using mpmath.quad for high precision.
    """
    # Set mpmath precision
    # mp.dps = 50 # Set desired decimal places precision

    # Define the integrand for mpmath
    # mpmath works with its own types
    j = mp.j # Imaginary unit
    pi = mp.pi
    m_mp = mp.mpf(m)
    u_s_mp = mp.mpf(u_scaled_val)

    if mp.almosteq(u_s_mp, 0):
        if m % 2 == 0:
            return mp.mpc(2.0 / (m + 1))
        else:
            return mp.mpc(0)

    integrand_mp = lambda t: mp.power(t, m_mp) * mp.exp(-j * 2 * pi * u_s_mp * t)

    try:
        # Perform high-precision numerical integration
        result_mp = mp.quad(integrand_mp, [-1, 1])
        return complex(result_mp) # Convert mpmath complex to Python complex
    except Exception as e:
        print(f"Error during mpmath.quad for m={m}, u_scaled={u_scaled_val}: {e}")
        return np.nan + np.nan * 1j

if __name__ == "__main__":
    # Set mpmath precision globally for the main block
    mp.dps = 50
    print(f"Using mpmath with precision (dps): {mp.dps}")

    # Failing cases identified from the scan
    test_cases = [
        (20, 1e-7),
        (20, 1e-5),
        # Add other cases if needed, e.g., where Taylor/Recursion switch happens
        (10, 0.016), # Case with high error previously
        (15, 1.6e-2), # Case with high error previously
        (5, 1e-3), # Case with high error previously
        (6, 1e-3), # Case with high error previously
        (2, 1e-7), # Check a low m, very small u_s case
    ]

    print("\n" + "="*60)
    print("Comparing Implementations with mpmath High-Precision Quad")
    print("="*60)
    print(f"{'m':<3} | {'u_scaled':<10} | {'Numeric (Ours)':<45} | {'SymPy':<45} | {'mpmath Quad':<45}")
    print("-"*155)

    for m_val, u_s_val in test_cases:
        u_s_val_np = np.longdouble(u_s_val) # Input for our func

        # 1. Our numeric implementation
        start_time = time.time()
        numeric_res = mft_numeric_test(m_val, u_s_val_np)
        numeric_time = time.time() - start_time
        numeric_complex = complex(numeric_res)

        # 2. SymPy implementation
        start_time = time.time()
        sympy_res = mft_symbolic_original(m_val, u_s_val_np)
        sympy_time = time.time() - start_time
        sympy_complex = complex(sympy_res) # Already complex

        # 3. mpmath high-precision quad
        start_time = time.time()
        mpmath_res = mft_mpmath_quad(m_val, u_s_val)
        mpmath_time = time.time() - start_time
        mpmath_complex = complex(mpmath_res) # Already complex

        # Format for printing
        numeric_str = f"{numeric_complex.real:+.6e}{numeric_complex.imag:+.6e}j (t={numeric_time:.2f}s)"
        sympy_str = f"{sympy_complex.real:+.6e}{sympy_complex.imag:+.6e}j (t={sympy_time:.2f}s)" if not np.isnan(sympy_complex.real) else "NaN"
        mpmath_str = f"{mpmath_complex.real:+.6e}{mpmath_complex.imag:+.6e}j (t={mpmath_time:.2f}s)" if not np.isnan(mpmath_complex.real) else "NaN"

        print(f"{m_val:<3} | {u_s_val:<10.3e} | {numeric_str:<45} | {sympy_str:<45} | {mpmath_str:<45}")

        # Calculate differences relative to mpmath
        diff_num_mp = np.abs(numeric_complex - mpmath_complex)
        diff_sym_mp = np.abs(sympy_complex - mpmath_complex)
        rel_diff_num_mp = diff_num_mp / (np.abs(mpmath_complex) + 1e-50) # Avoid div by zero
        rel_diff_sym_mp = diff_sym_mp / (np.abs(mpmath_complex) + 1e-50)

        print(f"{' ':>3} | {' ':<10} | Num vs MP: Abs={diff_num_mp:.3e}, Rel={rel_diff_num_mp:.3e}")
        print(f"{' ':>3} | {' ':<10} | Sym vs MP: Abs={diff_sym_mp:.3e}, Rel={rel_diff_sym_mp:.3e}")
        print("-"*155)

    print("\nVerification complete.") 