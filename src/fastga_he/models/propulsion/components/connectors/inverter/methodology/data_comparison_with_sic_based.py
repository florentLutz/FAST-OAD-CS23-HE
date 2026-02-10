#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import numpy as np

if __name__ == "__main__":
    # Those are straight up inputs of the sizing process
    gate_voltage_diode_semikron = 1.3
    gate_voltage_igbt_semikron = 1.3

    # So actually, it is higher
    gate_voltage_diode_sic_based = 4.2
    gate_voltage_igbt_sic_based = 4.3

    print(
        "Gate voltage diode, default vs SiC based:",
        gate_voltage_diode_semikron,
        gate_voltage_diode_sic_based,
    )
    print(
        "Gate voltage IGBT module, default vs SiC based:",
        gate_voltage_igbt_semikron,
        gate_voltage_igbt_sic_based,
    )

    # Rated current
    rated_current_ref_semikron = 450
    on_state_resistance_igbt_ref_semikron = 1.51e-3
    on_state_resistance_diode_ref_semikron = 1.87e-3
    rated_current_sic_based = 200.0

    on_state_resistance_igbt_sic_based = 2.9e-3
    # Assumed equal data missing
    on_state_resistance_diode_sic_based = 2.9e-3
    on_state_resistance_igbt_semikron = (
        rated_current_ref_semikron / rated_current_sic_based * on_state_resistance_igbt_ref_semikron
    )
    on_state_resistance_diode_semikron = (
        rated_current_ref_semikron
        / rated_current_sic_based
        * on_state_resistance_diode_ref_semikron
    )

    print(
        "On state resistance diode, default vs SiC based:",
        on_state_resistance_diode_semikron,
        on_state_resistance_diode_sic_based,
    )
    print(
        "Gate voltage IGBT module, default vs SiC based:",
        on_state_resistance_igbt_semikron,
        on_state_resistance_igbt_sic_based,
    )

    a_on_default = 0.015862856846788793 * rated_current_sic_based / rated_current_ref_semikron
    a_off_default = 0.004248159711027629 * rated_current_sic_based / rated_current_ref_semikron
    a_rr_default = 0.01507362815506554 * rated_current_sic_based / rated_current_ref_semikron

    b_on_default = 3.3256504177779456e-05
    b_off_default = 0.00034030311012866587
    b_rr_default = 0.0002539092853205021

    c_on_default = (
        5.134677955597825e-07 * (rated_current_sic_based / rated_current_ref_semikron) ** -1
    )
    c_off_default = (
        -4.5116538387667774e-08 * (rated_current_sic_based / rated_current_ref_semikron) ** -1
    )
    c_rr_default = (
        -1.739057882676467e-07 * (rated_current_sic_based / rated_current_ref_semikron) ** -1
    )

    current_for_energy_eval = 280.0
    e_on_semikron = (
        a_on_default / 2
        + b_on_default * current_for_energy_eval / np.pi
        + c_on_default * current_for_energy_eval**2.0 / 4
    )
    e_on_sic_based = 8e-3

    print(
        "Turn on energy losses, default vs SiC based:",
        e_on_semikron,
        e_on_sic_based,
        e_on_semikron / e_on_sic_based,
    )

    e_off_semikron = (
        a_off_default / 2
        + b_off_default * current_for_energy_eval / np.pi
        + c_off_default * current_for_energy_eval**2.0 / 4
    )
    e_off_sic_based = 7e-3

    print(
        "Turn off energy losses, default vs SiC based:",
        e_off_semikron,
        e_off_sic_based,
        e_off_semikron / e_off_sic_based,
    )

    e_rr_semikron = (
        a_rr_default / 2
        + b_rr_default * current_for_energy_eval / np.pi
        + c_rr_default * current_for_energy_eval**2.0 / 4
    )
    e_rr_sic_based = 3.1e-3
    print(
        "Reverse recovery energy losses, default vs SiC based:",
        e_rr_semikron,
        e_rr_sic_based,
        e_rr_semikron / e_rr_sic_based,
    )
