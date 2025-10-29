"""
Torque constant obtains from the SM PMSM for the medium term target of 2025 described in the
HASTECS  project :cite:`gudmundsson:2013`. The motor electromagnetic power is set to equal to the
shaft output as the mechanical losses is negligible.
"""

import numpy as np

if __name__ == "__main__":
    Pem_PU = 1.0  # Electromagnetic Power in PU
    S_base = 1432599.9999999997  # Base power of SM PMSM [W]
    RPM = 15970  # Rotational speed [rpm]
    Sp_power = 6300  # PMSM Specific power [W/kg]
    W = 115.4 + 33.8 + 56.7 + 19.7  # motor weight [kg]
    hs = 0.0358  # conductor slot height [m]
    ls = 0.0137  # conductor slot width [m]
    k_fill = 0.5  # conductor slot fill factor
    k_sc = 1  # Slot-conductor factor
    j_rms = 8.1e6  # RMS current density [A/m²]

    Power = Sp_power * W  # shaft_power_out
    Omega = 2 * np.pi * RPM / 60  # Mechanical angular speed [rad/s]
    t_em = Power / Omega  # torque
    print(f"Electromagnetic torque (t_em): {t_em:.4f} Nm")

    S_slot = hs * ls
    S_cond = S_slot * k_sc * k_fill
    num_phase = 3.0
    i_rms = S_cond * j_rms * num_phase
    print(f"RMS phase currnet (i_rms) : {i_rms:.4f} A")

    k_t = t_em / i_rms
    print(f"Torque constant (k_t): {k_t:.4f} Nm/A")
