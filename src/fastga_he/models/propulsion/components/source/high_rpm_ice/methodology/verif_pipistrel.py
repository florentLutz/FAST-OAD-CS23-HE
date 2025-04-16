# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

if __name__ == "__main__":
    altitude_validation_points = np.array(
        [
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            4000.0,
            4000.0,
            4000.0,
            4000.0,
            6000.0,
            6000.0,
            6000.0,
            8000.0,
            8000.0,
            8000.0,
            10000.0,
            10000.0,
            12000.0,
        ]
    )
    rpm_validation_points = np.array(
        [
            5500.0,
            5500.0,
            5300.0,
            4900.0,
            4600.0,
            5500.0,
            5500.0,
            5100.0,
            4600.0,
            5500.0,
            5300.0,
            4900.0,
            5500.0,
            5300.0,
            5100.0,
            5300.0,
            5500.0,
            5500.0,
        ]
    )
    power_validation_points = (
        np.array(
            [
                1.0,
                0.85,
                0.75,
                0.65,
                0.55,
                0.85,
                0.75,
                0.65,
                0.55,
                0.75,
                0.65,
                0.55,
                0.75,
                0.65,
                0.55,
                0.65,
                0.55,
                0.55,
            ]
        )
        * 69.0
    )  # Power is given as percentage of MCP
    map_validation_points = np.array(
        [
            27.7,
            26.7,
            25.7,
            24.7,
            24.0,
            25.3,
            24.3,
            23.3,
            23.3,
            23.3,
            22.7,
            22.0,
            22.0,
            21.7,
            21.0,
            19.7,
            20.3,
            18.0,
        ]
    )  # In inHg
    vol_ff_validation_points = np.array(
        [
            28.8,
            22.4,
            18.4,
            16.0,
            14.4,
            25.2,
            19.6,
            16.8,
            15.6,
            23.2,
            19.6,
            16.8,
            23.6,
            21.2,
            18.0,
            22.4,
            19.2,
            20.4,
        ]
    )  # In l/h

    displacement_volume = 1352.0  # In cm**3
    displacement_volume /= 1e6

    mep_validation_points = (
        (power_validation_points * 1000.0 * 2.0 * np.pi * 4.0)
        / (displacement_volume * rpm_validation_points * 2.0 * np.pi / 60.0)
        / 1e5
    )

    clipped_mep = np.clip(mep_validation_points, 5.0, None)
    sfc = 300.58 + np.exp(-0.44344488432412305 * (clipped_mep - 18.0))  # In g/kWh

    actual_sfc = vol_ff_validation_points * 0.72 * 1000 / power_validation_points

    # We'll verify only the cruise point and use that as the k factor

    mep_cruise = (
        0.75
        * 69
        * 1000.0
        * 2.0
        * np.pi
        * 4.0
        / (displacement_volume * 5300.0 * 2.0 * np.pi / 60.0)
        / 1e5
    )

    sfc_cruise = 300.58 + np.exp(-0.44344488432412305 * (mep_cruise - 18.0))  # In g/kWh
    actual_sfc_cruise = 18.4 * 0.72 * 1000.0 / (0.75 * 69.0)

    print(sfc_cruise)
    print(actual_sfc_cruise)

    print(actual_sfc_cruise / sfc_cruise)
