import jax
from scipy.optimize import fsolve


def equation_to_solve(corr_x, Mach, mflowratio):
    """Implicit equation: f(corr_x, Mach, mflowratio) = 0"""
    return (
        0.99575
        + 0.72927 * Mach**2
        + 34.61116 * corr_x
        - 36.33161 * corr_x**2
        + 154.13563 * corr_x**3 * Mach
        + 2.35051 * Mach**4
        - 3.67345 * Mach**3
        - 53.10867 * corr_x * Mach
        + 24.61205 * corr_x * Mach**3
        - mflowratio
    )


# Partial derivatives
df_dcorr_x = jax.grad(equation_to_solve, argnums=0)  # ∂f/∂corr_x
df_dmach = jax.grad(equation_to_solve, argnums=1)  # ∂f/∂Mach
df_dmflowratio = jax.grad(equation_to_solve, argnums=2)  # ∂f/∂mflowratio


def solve_with_sensitivities(Mach, mflowratio, initial_guess=0.1):
    """
    Solve for corr_x and compute its sensitivities to Mach and mflowratio

    Returns:
        corr_x: Output value
        dcorr_x_dMach: Sensitivity ∂corr_x/∂Mach
        dcorr_x_dmflowratio: Sensitivity ∂corr_x/∂mflowratio
    """

    # Solve for corr_x
    corr_x = fsolve(lambda x: equation_to_solve(x, Mach, mflowratio), initial_guess)[0]

    # Evaluate derivatives at solution point
    denom = float(df_dcorr_x(corr_x, Mach, mflowratio))
    numer_mach = float(df_dmach(corr_x, Mach, mflowratio))
    numer_mflowratio = float(df_dmflowratio(corr_x, Mach, mflowratio))

    # Implicit differentiation: dcorr_x/dMach = -(∂f/∂Mach)/(∂f/∂corr_x)
    dcorr_x_dMach = -numer_mach / denom
    dcorr_x_dmflowratio = -numer_mflowratio / denom

    return corr_x, dcorr_x_dMach, dcorr_x_dmflowratio


# Example
Mach = 0.8
mflowratio = 0.4

corr_x, d_mach, d_mflowratio = solve_with_sensitivities(Mach, mflowratio)

print(f"Inputs: Mach={Mach}, mflowratio={mflowratio}")
print(f"\nOutput: corr_x = {corr_x:.8f}")
print("\nSensitivities:")
print(f"  ∂corr_x/∂Mach = {d_mach:.6f}")
print(f"  ∂corr_x/∂mflowratio = {d_mflowratio:.6f}")
print("\nInterpretation:")
print(f"  If Mach increases by 0.01, corr_x changes by {d_mach * 0.01:.6f}")
print(f"  If mflowratio increases by 0.01, corr_x changes by {d_mflowratio * 0.01:.6f}")
