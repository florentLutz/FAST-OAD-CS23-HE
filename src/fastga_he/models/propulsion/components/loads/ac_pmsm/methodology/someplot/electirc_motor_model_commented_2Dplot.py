import numpy as np
import matplotlib.pyplot as plt

# Carica i coefficienti salvati (matrice 4x4)
coeffs_reshaped = np.load("coeffs_reshaped.npy")

if __name__ == "__main__":
    # Physical constants
    mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]
    Pem = 1432599.9999999997  # Base power [W]

    # Input parameters
    RPM = 15970  # Rotational speed [rpm]
    rho_cu_20 = 1.68e-8  # Copper resistivity at 20°C [Ohm·m]
    alpha_th = 0.00393  # Temperature coefficient for copper [1/°C]
    T_win = 180  # Winding temperature [°C]
    sigma = 50000  # Tangential stress [N/m²]
    j_rms = 8.1e6  # RMS current density [A/m²]
    A_rms = 81.4e3  # RMS linear current density
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
    # Air properties
    mu_air = 8.88e-15 * T**3 - 3.23e-11 * T**2 + 6.26e-8 * T + 2.35e-6
    rho_air = 1.293 * (273.15 / T) * pr

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

    Pmax_range = np.linspace(100000, Pem, 200)
    RPM_range = np.linspace(2000, 15970, 200)

    efficiency_grid = np.zeros((len(Pmax_range), len(RPM_range)))
    Sp_grid = np.zeros((len(Pmax_range), len(RPM_range)))

    # Ciclo su tutti i valori
    for i, Pem in enumerate(Pmax_range):
        for j, RPM in enumerate(RPM_range):
            # Derived quantities
            Omega = 2 * np.pi * RPM / 60  # Mechanical angular speed [rad/s]
            Torque = Pem / Omega

            # Equation II-43: Stator inner radius R
            R = ((lambda_ / (4 * np.pi * sigma)) * (Pem / Omega)) ** (1 / 3)
            R_r = x * R  # Rotor outer radius
            e_g = R - R_r  # Airgap thickness

            # Equation II-44: Active length Lm
            Lm = (2 / lambda_) * R
            L = Lm * k_tb

            # Equation II-45: Stator yoke thickness hy
            hy = (R / p) * np.sqrt(
                (B_m / B_sy) ** 2
                + (mu_0**2) * ((K_m / B_sy) ** 2) * (((1 + x_2p) / (1 - x_2p)) ** 2)
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

            # External stator radius
            R_out = R + hy + hs

            # Tooth to bore radius ratio
            term1 = (B_m / B_st) ** 2
            term2 = (mu_0**2) * ((K_m / B_st) ** 2) * ((1 + x_2p) / (1 - x_2p)) ** 2
            r_tooth = (2 / np.pi) * np.sqrt(term1 + term2)
            Ns = 2 * p * q * m
            ls = (1 - r_tooth) * 2 * np.pi * R / Ns
            # ls = 0.01348

            # Frame radius
            R_sh = R_r / 3
            R_out_mm = R_out * 1000

            tau_r_ = 0.7371 * R_out**2 - 0.580 * R_out + 1.1599 if R_out_mm <= 400 else 1.04
            R_fr = tau_r_ * R_out

            # Stator weight
            W_stat_core = (np.pi * Lm * (R_out**2 - R**2) - (hs * Lm * Ns * ls)) * rho_sf
            vol_wind = k_tb * k_lc * hs * Lm * Ns * ls
            mat_mix_density = k_fill * rho_c + (1 - k_fill) * rho_insl
            W_stat_wind = vol_wind * mat_mix_density
            W_stat = W_stat_core + W_stat_wind

            # Rotor density based on pole pairs
            if p <= 10:
                rho_rot = -431.67 * p + 7932
            elif 10 < p <= 50:
                rho_rot = 1.09 * p**2 - 117.45 * p + 4681
            else:
                rho_rot = 1600

            # Rotor weight
            W_rot = np.pi * R_r**2 * Lm * rho_rot

            # Frame weight
            W_frame = rho_fr * (
                np.pi * Lm * k_tb * (R_fr**2 - R_out**2)
                + 2 * np.pi * (tau_r_ - 1) * R_out * R_fr**2
            )

            # Total motor weight
            W_mot = W_stat + W_rot + W_frame

            ########################## LOSSES ##########################
            # Joule losses
            kk_t = 0.4278  # Nm/A  torque constant
            N_c = 2 * p * q * m
            l_c = Lm * k_lc * k_tb
            S_slot = hs * ls
            S_cond = S_slot * k_sc * k_fill
            rho_cu_Twin = rho_cu_20 * (1 + alpha_th * (T_win - 20))
            Resistance = N_c * rho_cu_Twin * l_c / S_cond
            I_rms = Torque / kk_t
            P_j = Resistance * I_rms**2
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

            # Rotor windage
            Re_rot = (rho_air * R_r**2 * Omega) / mu_air
            C_fr = 3.87 / Re_rot**0.5 if Re_rot < 3e5 else 0.146 / Re_rot**0.2
            P_windage_rotor = 0.5 * C_fr * np.pi * rho_air * Omega**3 * (R_r**5 - R_sh**5)

            # Bearing friction losses
            P_eq = W_rot * g
            P_friction = 0.5 * Cf_bearing * P_eq * d_bearing * Omega

            # Total mechanical losses
            P_mec_loss = (P_windage_airgap + 2 * P_windage_rotor) + (2 * P_friction)

            # Iron losses
            # Valori di test
            f = RPM / 30  # Frequenza in Hz

            sqrt_f = np.sqrt(f)
            sqrt_Bm = np.sqrt(B_m)

            P_iron = 0

            for k in range(4):  # k = potenza di sqrt(f)
                for b in range(4):  # b = potenza di sqrt(Bm)
                    coeff = coeffs_reshaped[k][b]
                    term = coeff * (sqrt_f ** (k + 1)) * (sqrt_Bm ** (b + 1))
                    P_iron += term

            # Se necessario, moltiplica per un fattore di scala (es. 224.88)
            P_iron *= W_mot

            P_loss = P_j + P_iron + P_mec_loss
            #
            Pele = Pem + 2 * P_loss  # + 2 * (P_j + P_iron)
            # Mechanical output power
            Pmec = Pem

            # Efficiency
            efficiency_grid[i, j] = Pmec / Pele

            # Specific power
            Sp_grid[i, j] = Pem / W_mot


# Plot della superficie
RPM_grid, P_grid = np.meshgrid(RPM_range, Pmax_range)

plt.figure(figsize=(10, 6))
cp2 = plt.contourf(RPM_grid, P_grid, efficiency_grid, levels=20, cmap="viridis")
plt.colorbar(cp2, label="Efficiency")
plt.xlabel("RPM [W]")
plt.ylabel("Power [kW]")
plt.title("Efficiency vs RPM and Power")
plt.tight_layout()
plt.show()


S_grid, R_grid = np.meshgrid(RPM_range, Pmax_range)

plt.figure(figsize=(10, 6))
cp = plt.contourf(RPM_grid, P_grid, Sp_grid, levels=20, cmap="viridis")
plt.colorbar(cp, label="Specific Power [W/kg]")
plt.xlabel("RPM [1/min]")
plt.ylabel("Power [W]")
plt.title("Specific Power  vs RPM and Power")
plt.tight_layout()
plt.show()


i = 100  # Ultimo valore disponibile
Sp_vec = Sp_grid[i, :]
Torque_fixed = Pmax_range[i]

plt.figure(figsize=(10, 6))
plt.plot(RPM_range, Sp_vec, label=f"Torque = {Torque_fixed:.1f} Nm", color="darkorange")
plt.xlabel("RPM [1/min]")
plt.ylabel("Specific Power [W/kg]")
plt.title("Specific Power vs RPM at fixed Torque")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
