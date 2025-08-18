import numpy as np

""" PMSM model from HASTECS project, Sarah Touhami"""
if __name__ == "__main__":
    # Physical constants
    mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]

    # Input parameters
    Pem_PU = 1.0  # Power in PU
    S_base = 1432599.9999999997  # 1432599.9999999997  # Base power [W]
    RPM = 15970  # Rotational speed [rpm]
    rho_cu_20 = 1.68e-8  # Copper resistivity at 20°C [Ohm·m]
    alpha_th = 0.00393  # Temperature coefficient for copper [1/°C]
    T_win = 180  # Winding temperature [°C]
    sigma = 50000  # Tangential stress [N/m²]
    j_rms = 8.1e6  # RMS current density [A/m²]
    A_rms = 81.4e3  # RMS linear current density [A/m]
    K_m = 111.100  # Max surface current density [A/m]
    B_m = 0.9  # Airgap flux density [T]
    B_st = 1.3  # Tooth flux density [T]
    B_sy = 1.2  # Yoke flux density [T]
    lambda_ = 0.6  # Form coefficient
    k_fill = 0.5  # Slot fill factor
    p = 2  # Number of pole pairs
    k_tb = 1.4  # End winding coefficient

    # Unknown constants (assumed)
    k_w = 0.97  # Winding factor
    k_sc = 1  # Slot-conductor factor
    x = 0.97  # Rotor/stator radius ratio
    k_lc = 1.25  # conductor twisting coefficient
    x_2p = x ** (2 * p)
    T = 300  # Air temperature [K]
    k1 = 1  # Smoothness factor
    pr = 1  # Air pressure [atm]

    # Constants and material properties
    Cf_bearing = 0.0015  # Bearing friction coefficient
    d_bearing = 0.03  # Bearing bore diameter [m]
    g = 9.81  # Gravity [m/s²]

    # Motor configuration
    q = 3  # Number of phases
    m = 2  # Slots per pole per phase
    # ls = 0.01348                           # Stator slot width [m]

    # Material densities
    rho_sf = 8150  # Soft magnetic material density [kg/m³]
    rho_c = 8960  # Copper density [kg/m³]
    rho_insl = 1400  # Insulation density [kg/m³]
    rho_fr = 2100  # Frame material density [kg/m³]

    # Derived quantities
    Pem = Pem_PU * S_base  # Real power [W]
    Omega = 2 * np.pi * RPM / 60  # Mechanical angular speed [rad/s]
    Torque = Pem / Omega
    print(f"tORQUE (T): {Torque:.4f} Nm")
    # Equation II-43: Stator inner radius R
    R = ((lambda_ / (4 * np.pi * sigma)) * (Pem / Omega)) ** (1 / 3)
    R_r = x * R  # Rotor outer radius
    e_g = R - R_r  # Airgap thickness
    print(f"Airgap thickness (e_g): {e_g:.4f} m")

    # Equation II-44: Active length Lm
    Lm = (2 / lambda_) * R
    L = Lm * k_tb

    # Equation II-45: Stator yoke thickness hy
    hy = (R / p) * np.sqrt(
        (B_m / B_sy) ** 2 + (mu_0**2) * ((K_m / B_sy) ** 2) * (((1 + x_2p) / (1 - x_2p)) ** 2)
    )

    # Equation II-46: Slot height hs
    numerator = np.sqrt(2) * sigma
    denominator = k_w * B_m * j_rms
    second_term = 1 / (
        k_sc
        * k_fill
        * (
            1
            - (
                (2 / np.pi)
                * np.sqrt(
                    (B_m / B_st) ** 2
                    + ((mu_0 * K_m / B_st) ** 2) * (((1 + x_2p) / (1 - x_2p)) ** 2)
                )
            )
        )
    )
    hs = (numerator / denominator) * second_term

    # Display core geometry results
    print(f"Stator inner radius (R): {R:.4f} m")
    print(f"Active length (Lm): {Lm:.4f} m")
    print(f"Stator yoke thickness (hy): {hy:.4f} m")
    print(f"Slot height (hs): {hs:.4f} m")

    # External stator radius
    R_out = R + hy + hs

    # Tooth to bore radius ratio
    term1 = (B_m / B_st) ** 2
    term2 = (mu_0**2) * ((K_m / B_st) ** 2) * ((1 + x_2p) / (1 - x_2p)) ** 2
    r_tooth = (2 / np.pi) * np.sqrt(term1 + term2)
    Ns = 2 * p * q * m
    ls = (1 - r_tooth) * 2 * np.pi * R / Ns
    # ls = 0.01348

    print(f"Stator outer radius (R_out): {R_out:.4f} m")
    print(f"Stator slot width (ls): {ls:.4f} m")

    # Frame radius
    R_sh = R_r / 3
    R_out_mm = R_out * 1000

    tau_r_ = 0.7371 * R_out**2 - 0.580 * R_out + 1.1599 if R_out_mm <= 400 else 1.04
    R_fr = tau_r_ * R_out
    print(f"Frame radius (R_out): {R_fr:.4f} m")

    # Stator weight
    W_stat_core = (np.pi * Lm * (R_out**2 - R**2) - (hs * Lm * Ns * ls)) * rho_sf
    vol_wind = k_tb * k_lc * hs * Lm * Ns * ls
    mat_mix_density = k_fill * rho_c + (1 - k_fill) * rho_insl
    W_stat_wind = vol_wind * mat_mix_density
    W_stat = W_stat_core + W_stat_wind

    print(f"Stator core weight: {W_stat_core:.2f} kg")
    print(f"Stator winding weight: {W_stat_wind:.2f} kg")
    print(f"Total stator weight: {W_stat:.2f} kg")

    # Rotor density based on pole pairs
    if p <= 10:
        rho_rot = -431.67 * p + 7932
    elif 10 < p <= 50:
        rho_rot = 1.09 * p**2 - 117.45 * p + 4681
    else:
        rho_rot = 1600

    # Rotor weight
    W_rot = np.pi * R_r**2 * Lm * rho_rot
    print(f"Rotor weight: {W_rot:.2f} kg")

    # Frame weight
    W_frame = rho_fr * (
        np.pi * Lm * k_tb * (R_fr**2 - R_out**2) + 2 * np.pi * (tau_r_ - 1) * R_out * R_fr**2
    )
    print(f"Frame weight: {W_frame:.2f} kg")

    # Total motor weight
    W_mot = W_stat + W_rot + W_frame
    print(f"Total motor weight: {W_mot:.2f} kg")

    ########################## LOSSES ##########################
    # Joule losses
    rho_cu_Twin = rho_cu_20 * (1 + alpha_th * (T_win - 20))
    P_j = rho_cu_Twin * k_tb * k_lc * Lm * (2 * np.pi * R * A_rms * j_rms)
    print(f"Joule losses: {P_j:.2f} W")

    # Air properties
    mu_air = 8.88e-15 * T**3 - 3.23e-11 * T**2 + 6.26e-8 * T + 2.35e-6
    rho_air = 1.293 * (273.15 / T) * pr

    # Reynolds number and friction coefficient
    Re_a = (rho_air * R_r * e_g * Omega) / mu_air
    if 500 < Re_a < 1e4:
        Cf_a = 0.515 * (e_g / R_r) ** 0.3 * (Re_a**-0.5)
    elif Re_a >= 1e4:
        Cf_a = 0.0325 * (e_g / R_r) ** 0.3 * (Re_a**-0.2)
    else:
        raise ValueError("Re_a is too low (Re_a < 500)")

    P_windage_airgap = k1 * Cf_a * np.pi * rho_air * R_r**4 * Omega**3 * L
    print(f"Airgap windage losses: {P_windage_airgap:.2f} W")

    # Rotor windage
    Re_rot = (rho_air * R_r**2 * Omega) / mu_air
    C_fr = 3.87 / Re_rot**0.5 if Re_rot < 3e5 else 0.146 / Re_rot**0.2
    P_windage_rotor = 0.5 * C_fr * np.pi * rho_air * Omega**3 * (R_r**5 - R_sh**5)
    print(f"Rotor windage losses: {P_windage_rotor:.2f} W")

    # Bearing friction losses
    P_eq = W_rot * g
    P_friction = 0.5 * Cf_bearing * P_eq * d_bearing * Omega
    print(f"Bearing friction losses: {P_friction:.2f} W")

    # Total mechanical losses
    P_mec_loss = (P_windage_airgap + 2 * P_windage_rotor) + (2 * P_friction)
    print(f"Total mechanical losses: {P_mec_loss:.2f} W")

    # Iron losses
    # Carica i coefficienti salvati (matrice 4x4)
    coeffs_reshaped = np.load("coeffs_reshaped.npy")

    # Valori di test
    f = RPM / 30  # Frequenza in Hz

    sqrt_f = np.sqrt(f)
    sqrt_Bm = np.sqrt(B_m)

    P_iron = 0

    for i in range(4):  # i = potenza di sqrt(f)
        for j in range(4):  # j = potenza di sqrt(Bm)
            coeff = coeffs_reshaped[i][j]
            term = coeff * (sqrt_f ** (i + 1)) * (sqrt_Bm ** (j + 1))
            P_iron += term

    # Se necessario, moltiplica per un fattore di scala (es. 224.88)
    P_iron *= 224.88

    print(f"Per f = {f:.2f} Hz e Bm = {B_m} T, le perdite stimate sono: {P_iron:.2f} W/kg")

    P_loss = P_j + P_mec_loss + P_iron
    #
    Pele = Pem + 2 * P_loss  # + 2 * (P_j + P_iron)
    # Mechanical output power
    Pmec = Pem  # P_mec_loss
    print(f"Mechanical output power: {Pmec:.2f} W")

    # Efficiency
    efficiency = Pmec / Pele
    print(f"Efficiency: {efficiency:.4f}")
    # Specific power
    Sp = Pem / W_mot
    print(f"Specific power: {Sp:.2f} W/kg")

    # Calculation of maximum electromagnetic torque
    S_rot = 2 * np.pi * R_r * Lm  # Rotor surface area
    Tem_max = Pem / Omega  # Maximum electromagnetic torque

    # Specific torque
    ST = Tem_max / W_mot

    # Output
    print(f"Maximum electromagnetic torque (Tem_max) = {Tem_max:.2f} Nm")
    print(f"Specific torque (ST) = {ST:.2f} Nm/kg")

    A_jeq = (P_j + P_mec_loss + P_iron) / (2 * np.pi * R * k_tb * Lm * rho_cu_Twin)

    print(f"A_jeq = {A_jeq:.2f} ")
