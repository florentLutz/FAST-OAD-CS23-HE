import numpy as np

if __name__ == "__main__":
    Pem_PU = 1.0  # Power in PU
    S_base = 1432599.9999999997  # Base power [W]
    RPM = 15970  # Rotational speed [rpm]
    Sp_power = 6300  # Specific power [W/kg]
    W = 115.4 + 33.8 + 56.7 + 19.7  # motor weight [kg]
    hs = 0.0358  # slot height [m]
    ls = 0.0137  # Stator slot width [m]
    k_fill = 0.5  # Slot fill factor
    k_sc = 1  # Slot-conductor factor
    j_rms = 8.1e6  # RMS current density [A/m²]

    Power = Sp_power * W  # shaft_power_out
    Omega = 2 * np.pi * RPM / 60  # Mechanical angular speed [rad/s]
    T_em = Power / Omega  # torque
    print(f"T_em: {T_em:.4f} Nm")

    S_slot = hs * ls
    S_cond = S_slot * k_sc * k_fill
    I_rms = S_cond * j_rms
    print(f"I_rms: {I_rms:.4f} Nm/A")
    S_cond = I_rms / j_rms
    S_slot = S_cond / (k_sc * k_fill)
    hs = S_slot / ls
    print(f"hs: {hs:.4f} m")

    k_t = T_em / I_rms
    print(f"k_t: {k_t:.4f} Nm/A")
