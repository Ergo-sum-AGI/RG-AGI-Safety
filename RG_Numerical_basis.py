#!/usr/bin/env python3
"""
rg_phi_d3.py
Bubble integral + beta-function fixed point solver (d=3).
Includes convergence scan for residuals and stability.
Daniel Solis – 2025-07-10
"""

from mpmath import mp, pi, gamma, quad, findroot, diff

# ------------------------------------------------------
# Precision setup
# ------------------------------------------------------
mp.dps = 80

# ------------------------------------------------------
# Geometry factors
# ------------------------------------------------------
def omega_d(d):
    """Surface area of unit sphere in d dimensions."""
    return 2 * mp.pi**(d/2) / mp.gamma(d/2)

# ------------------------------------------------------
# Bubble integral (analytic + numeric)
# ------------------------------------------------------
def I_analytic(alpha, d=3, Lambda=1.0, b=1.1):
    """Analytic one-loop bubble integral."""
    pref = omega_d(d) / (2 * mp.pi)**d
    exponent = d - 2 - alpha
    if abs(exponent) < mp.mpf('1e-30'):
        return pref * mp.log(b)
    else:
        return pref * (Lambda**exponent - (Lambda/b)**exponent) / exponent

def I_numeric(alpha, d=3, Lambda=1.0, b=1.1):
    """Numerical quadrature check."""
    lower, upper = Lambda/b, Lambda
    integrand = lambda q: q**(d - 3 - alpha)
    val = quad(integrand, [lower, upper])
    return (omega_d(d) / (2 * mp.pi)**d) * val

# ------------------------------------------------------
# Beta-function for d=3
# ------------------------------------------------------
def f_d3(alpha):
    """Bracket factor f(α) for d=3."""
    return 1 - (1 + alpha)/2 * (gamma(alpha/2) / gamma((3 - alpha)/2))

def beta_d3(alpha):
    """β(α) = (α - 4) f(α) for d=3."""
    return (alpha - 4) * f_d3(alpha)

# ------------------------------------------------------
# Fixed-point coupling g*(α)
# ------------------------------------------------------
def g_star(alpha, d=3):
    """Fixed-point coupling g*(α) with full prefactors."""
    Omega_d = omega_d(d)
    return (2 * pi)**d / Omega_d * (d - 2 - alpha) * (alpha - 4)

# ------------------------------------------------------
# Convergence scan
# ------------------------------------------------------
def convergence_scan():
    print("\n=== Convergence, Residual, and Stability Check (d=3) ===")
    for dps in [60, 80, 100]:
        mp.dps = dps
        try:
            root = findroot(f_d3, mp.mpf('1.618'))
            fval = f_d3(root)
            beta_val = beta_d3(root)
            beta_prime = diff(beta_d3, root)
            gstar = g_star(root, d=3)
            print(f"dps={dps:<3}  α*={root}  f(α*)={fval:.2e}  β(α*)={beta_val:.2e}  β′(α*)={beta_prime:.6f}  g*={gstar}")
        except Exception as e:
            print(f"dps={dps:<3}  ERROR: {e}")

# ------------------------------------------------------
# Main
# ------------------------------------------------------
if __name__ == "__main__":
    d = 3
    print("=== Bubble Integral Evaluation (d=3) ===")
    for alpha in [1.5, 1.6180339887, 2.0]:
        Ia = I_analytic(alpha, d=d, Lambda=1.0, b=1.2)
        In = I_numeric(alpha, d=d, Lambda=1.0, b=1.2)
        rel = abs((Ia - In)/Ia) if Ia != 0 else abs(In)
        print(f"α={alpha:.12f}  I_analytic={Ia:.12e}  I_numeric={In:.12e}  rel.err={rel:.2e}")

    print("\n=== Beta-function Root Solve (d=3) ===")
    root = findroot(f_d3, mp.mpf('1.618'))
    fval = f_d3(root)
    beta_val = beta_d3(root)
    beta_prime = diff(beta_d3, root)
    gstar = g_star(root, d=3)

    print(f"α* = {root}")
    print(f"f(α*) = {fval}")
    print(f"β(α*) = {beta_val}")
    print(f"β′(α*) = {beta_prime}   (negative => IR-attractive)")
    print(f"g*(α*) = {gstar}")

    # Run convergence certificate
    convergence_scan()