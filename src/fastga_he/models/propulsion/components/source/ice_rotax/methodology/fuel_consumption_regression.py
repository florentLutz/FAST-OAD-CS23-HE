# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

if __name__ == "__main__":
    """
    This files presents how the regression for the fuel consumption of a rotax type engine were
    obtained. Data are based on the technical documentation of rotax engine obtained on the 
    following link: https://www.flyrotax.com/fr/p/services/documentation-techniques
    Accessed: 2025/03/25 10.43
    
    912 Series contains data for 912-A, -F, -S, -UL, -ULS
    914 Series contains data for 914-F, -UL
    """

    # Data for the 912 Series -A, -F, -UL
    data_rpm_power_912_a = np.array(
        [2499.0, 2996.0, 3494.0, 4001.0, 4492.0, 4993.0, 5490.0, 5794.0]
    )
    data_power_power_912_a = np.array([4.91, 8.60, 13.2, 19.9, 27.8, 38.3, 50.7, 59.7])

    data_rpm_fuel_912_a = np.array(
        [2494.83, 2997.04, 3490.39, 3995.56, 4497.78, 4997.04, 5502.21, 5794.68]
    )
    data_fuel_fuel_912_a = np.array([2.99, 4.16, 5.68, 8.14, 11.1, 14.2, 18.7, 22.4])

    # Data for the 912 Series -S, -ULS
    data_rpm_power_912_s = np.array(
        [2498.2, 2999.6, 3495.2, 3992.9, 4486.8, 4980.3, 5478.7, 5772.6]
    )
    data_power_power_912_s = np.array([6.4, 11.3, 16.3, 24.3, 35.1, 47.0, 62.5, 73.2])

    data_rpm_fuel_912_s = np.array([2496.2, 2993.9, 3491.3, 3989.4, 4487.5, 4985.4, 5482.0, 5778.1])
    data_fuel_fuel_912_s = np.array([5.99, 6.90, 8.17, 12.0, 15.9, 20.0, 25.5, 26.9])

    # Data for the 914 Series
    data_rpm_power_914 = np.array(
        [
            2493.0,
            2794.0,
            3002.0,
            3212.0,
            3497.0,
            3561.0,
            3844.0,
            4007.0,
            4088.0,
            4307.0,
            4507.0,
            4696.0,
            4865.0,
            5018.0,
            5174.0,
            5313.0,
            5438.0,
            5521.0,
            5705.0,
            5819.0,
        ]
    )
    data_power_power_914 = np.array(
        [
            7.14,
            9.90,
            12.3,
            15.0,
            19.0,
            19.9,
            25.1,
            28.0,
            30.0,
            35.1,
            40.0,
            45.1,
            50.1,
            54.9,
            60.2,
            65.1,
            70.2,
            73.2,
            80.4,
            84.9,
        ]
    )  # In kW

    data_rpm_fuel_914 = np.array([2492.0, 3000, 3495.0, 4000, 4495.0, 5000, 5501.0])
    data_fuel_fuel_914 = np.array(
        [
            6.08,
            7.62,
            10.3,
            14.5,
            20.1,
            26.2,
        ]
    )  # In l/h
