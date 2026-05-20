"""
PMSM Weibull Reliability Analysis
================================================================
Fits a 2-parameter Weibull model to B1/B10/B50 data points via
linearised least-squares regression :cite:`chen:2024`, then extrapolates to find the
operating lifespan corresponding to a 10⁻⁹ failure probability
(required for critical aviation applications).

B-points (from graph digitisation):
    B1  → t = 102,623 h  (1%  failure, R = 0.99)
    B10 → t = 140,757 h  (10% failure, R = 0.90)
    B50 → t = 210,348 h  (50% failure, R = 0.50)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

if __name__ == "__main__":
    # ── 1. Input data ────────────────────────────────────────────────────────────

    b_points = {
        "B₁":  {"t": 102_623, "R": 0.99},
        "B₁₀": {"t": 140_757, "R": 0.90},
        "B₅₀": {"t": 210_348, "R": 0.50},
    }

    # ── 2. Linearised Weibull regression ─────────────────────────────────────────
    #   ln(-ln(R)) = β·ln(t) - β·ln(η)   →   y = slope·x + intercept

    t_arr = np.array([v["t"] for v in b_points.values()], dtype=float)
    R_arr = np.array([v["R"] for v in b_points.values()], dtype=float)

    x = np.log(t_arr)
    y = np.log(-np.log(R_arr))

    # Ordinary least squares
    slope, intercept = np.polyfit(x, y, 1)
    beta = slope                          # Weibull shape parameter
    eta  = np.exp(-intercept / beta)      # Weibull scale parameter (characteristic life)

    # R² of the linearised fit
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print("=" * 55)
    print("  Weibull regression results")
    print("=" * 55)
    print(f"  Shape  β  = {beta:.4f}")
    print(f"  Scale  η  = {eta:,.0f} h  ({eta/1e5:.4f} × 10⁵ h)")
    print(f"  R²        = {r_squared:.8f}")
    print()

    # ── 3. Aviation lifespan targets ─────────────────────────────────────────────

    aviation_targets = {
        "Commercial (10⁻⁷)":        1e-7,
        "Military (10⁻⁸)":          1e-8,
        "Critical aviation (10⁻⁹)": 1e-9,
        "Ultra-critical (10⁻¹⁰)":   1e-10,
    }

    print("  Aviation lifespan targets")
    print("-" * 55)
    for label, fp in aviation_targets.items():
        R_target = 1 - fp
        t_target = eta * (-np.log(R_target)) ** (1 / beta)
        print(f"  {label:<28}  t = {t_target:,.0f} h")
    print("=" * 55)

    # Primary target
    fp_primary  = 1e-9
    R_primary   = 1 - fp_primary
    t_primary   = eta * (-np.log(R_primary)) ** (1 / beta)

    # ── 4. Weibull curve ─────────────────────────────────────────────────────────

    t_plot = np.linspace(1, 5e5, 2000)
    R_plot = np.exp(-(t_plot / eta) ** beta)

    # ── 5. Plot ───────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PMSM Weibull Reliability Analysis — Aviation Application",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── 5a. Main reliability curve ───────────────────────────────────────────────
    ax = axes[0]

    ax.plot(t_plot / 1e5, R_plot, color="#1a5fa8", linewidth=2.0, label="R(t) — Weibull fit")

    # B-points
    for label, d in b_points.items():
        ax.plot(d["t"] / 1e5, d["R"], "o", color="#d63031", markersize=7, zorder=5)
        ax.annotate(label, xy=(d["t"] / 1e5, d["R"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color="#d63031")
        # Dashed drop lines
        ax.plot([d["t"] / 1e5, d["t"] / 1e5], [0, d["R"]],
                "--", color="#d63031", linewidth=0.7, alpha=0.5)
        ax.plot([0, d["t"] / 1e5], [d["R"], d["R"]],
                "--", color="#d63031", linewidth=0.7, alpha=0.5)

    # Aviation target marker (t_primary is very small → near origin)
    ax.axvline(t_primary / 1e5, color="#27ae60", linewidth=1.5,
               linestyle=":", label=f"10⁻⁹ target  t = {t_primary:,.0f} h")
    ax.plot(t_primary / 1e5, R_primary, "^", color="#27ae60",
            markersize=9, zorder=6)

    ax.set_xlabel("t  /  h  (×10⁵)", fontsize=11)
    ax.set_ylabel("R(t)", fontsize=11)
    ax.set_title("Reliability curve R(t)", fontsize=11)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    legend_handles = [
        Line2D([0], [0], color="#1a5fa8", linewidth=2,   label="R(t) — Weibull fit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d63031",
               markersize=7, label="B₁ / B₁₀ / B₅₀ points"),
        Line2D([0], [0], color="#27ae60", linewidth=1.5, linestyle=":",
               marker="^", markerfacecolor="#27ae60",
               label=f"10⁻⁹ target  t ≈ {t_primary:,.0f} h"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    # Weibull parameters text box
    param_text = (f"β  = {beta:.4f}\n"
                  f"η  = {eta/1e5:.4f} × 10⁵ h\n"
                  f"R² = {r_squared:.8f}")
    ax.text(0.97, 0.55, param_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8),
            fontfamily="monospace")

    # ── 5b. Linearised Weibull (probability paper) ───────────────────────────────
    ax2 = axes[1]

    x_line = np.linspace(x.min() - 0.3, x.max() + 0.5, 100)
    y_line = slope * x_line + intercept

    ax2.plot(x_line, y_line, color="#1a5fa8", linewidth=1.8,
             label=f"Regression  β={beta:.3f}")
    ax2.scatter(x, y, color="#d63031", s=60, zorder=5, label="B-points")

    for i, (label, d) in enumerate(b_points.items()):
        ax2.annotate(label, xy=(x[i], y[i]),
                     xytext=(6, 4), textcoords="offset points",
                     fontsize=9, color="#d63031")

    # Aviation target on linearised scale
    y_aviation = np.log(-np.log(R_primary))
    x_aviation = np.log(t_primary)
    ax2.scatter([x_aviation], [y_aviation], color="#27ae60", s=80,
                marker="^", zorder=6, label="10⁻⁹ target")
    ax2.axvline(x_aviation, color="#27ae60", linewidth=1.2, linestyle=":", alpha=0.7)
    ax2.axhline(y_aviation, color="#27ae60", linewidth=1.2, linestyle=":", alpha=0.7)

    ax2.set_xlabel("ln(t)", fontsize=11)
    ax2.set_ylabel("ln(−ln(R))", fontsize=11)
    ax2.set_title("Linearised Weibull", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9)

    # Equation annotation
    eq_text = f"ln(−ln(R)) = {beta:.4f}·ln(t)  {intercept:+.4f}"
    ax2.text(0.03, 0.97, eq_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.8),
             fontfamily="monospace")

    plt.tight_layout()
    plt.show()
