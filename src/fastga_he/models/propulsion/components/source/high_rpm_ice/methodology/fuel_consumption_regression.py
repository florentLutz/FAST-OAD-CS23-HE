# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go


def mean_effective_pressure(displacement_volume, power, omega):
    mep = (power * 2.0 * np.pi * 4.0) / (displacement_volume * omega)
    return mep


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
        [2499.0, 2996.0, 3494.0, 4001.0, 4492.0, 4993.0, 5490.0, 5794.0]  # In kW
    )
    data_power_power_912_a = np.array([4.91, 8.60, 13.2, 19.9, 27.8, 38.3, 50.7, 59.7])

    data_rpm_fuel_912_a = np.array([2494.83, 2997.04, 3490.39, 3995.56, 4497.78, 4997.04, 5502.21])
    data_fuel_fuel_912_a = np.array([2.99, 4.16, 5.68, 8.14, 11.1, 14.2, 18.7])  # In l/h

    data_rpm_power_912_a_prop_curve = np.array(
        [
            2509.5238095238096,
            3007.936507936508,
            3496.8253968253966,
            3995.2380952380954,
            4500,
            4995.238095238095,
            5490.47619047619,
            5800.0,
        ]  # In rpm
    )
    data_power_power_912_a_prop_curve = np.array(
        [
            4.679144385026737,
            8.529157117392405,
            13.420422714540365,
            19.99770817417876,
            28.34071810542399,
            38.687547746371266,
            51.280366692131395,
            60.1,
        ]
    )

    data_rpm_power_912_a_max_curve = np.array(
        [
            2512.6984126984125,
            3007.9365079365075,
            3500,
            4004.7619047619046,
            4496.825396825396,
            5001.587301587301,
            5484.126984126984,
            5800.0,
        ]  # In rpm
    )
    data_power_power_912_a_max_curve = np.array(
        [
            21.28393175451999,
            28.50241914947797,
            34.83804430863254,
            41.57677616501145,
            47.992615227909326,
            54.16984975808504,
            58.73924115100585,
            60.1,
        ]
    )

    mid_curve_912_a_rpm = np.array([2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 5800.0])
    mid_curve_912_a_power = 0.5 * np.interp(
        mid_curve_912_a_rpm, data_rpm_power_912_a_prop_curve, data_power_power_912_a_prop_curve
    ) + 0.5 * np.interp(
        mid_curve_912_a_rpm, data_rpm_power_912_a_max_curve, data_power_power_912_a_max_curve
    )

    # Data for the 912 Series -S, -ULS
    data_rpm_power_912_s = np.array(
        [2498.2, 2999.6, 3495.2, 3992.9, 4486.8, 4980.3, 5478.7, 5772.6]  # In kW
    )
    data_power_power_912_s = np.array([6.4, 11.3, 16.3, 24.3, 35.1, 47.0, 62.5, 73.2])

    data_rpm_power_912_s_prop_curve = np.array(
        [
            2497.5094768070026,
            2996.493123592213,
            3501.648140348307,
            3997.8665738824693,
            4500.549380116103,
            5003.305437031882,
            5512.580804658744,
            5800,
        ]  # In rpm
    )
    data_power_power_912_s_prop_curve = np.array(
        [
            5.810060981192882,
            10.839635944109723,
            16.41035032138737,
            24.313183291519362,
            35.63151244345957,
            47.66843078726164,
            63.65978720676827,
            74.5,
        ]
    )

    data_rpm_power_912_s_max_curve = np.array(
        [
            2496.5022799274816,
            2995.797242111817,
            3501.208636255425,
            3994.4054791510243,
            4502.948339956416,
            5008.414672111635,
            5510.419909535407,
            5800,
        ]  # In rpm
    )
    data_power_power_912_s_max_curve = np.array(
        [
            25.929459593092464,
            34.01303862142214,
            42.09881517021625,
            50.359843976046996,
            59.16530847693516,
            67.79002691962563,
            72.4614060468437,
            74.5,
        ]
    )

    # Take the avg
    mid_curve_912_s_rpm = np.array([2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 5800.0])
    mid_curve_912_s_power = 0.5 * np.interp(
        mid_curve_912_s_rpm, data_rpm_power_912_s_prop_curve, data_power_power_912_s_prop_curve
    ) + 0.5 * np.interp(
        mid_curve_912_s_rpm, data_rpm_power_912_s_max_curve, data_power_power_912_s_max_curve
    )

    fig_mid_curve = go.Figure()
    fig_mid_curve.add_trace(
        go.Scatter(
            x=data_rpm_power_912_s_prop_curve,
            y=data_power_power_912_s_prop_curve,
            name="Prop curve",
        )
    )
    fig_mid_curve.add_trace(
        go.Scatter(
            x=data_rpm_power_912_s_max_curve, y=data_power_power_912_s_max_curve, name="Max curve"
        )
    )
    fig_mid_curve.add_trace(
        go.Scatter(x=mid_curve_912_s_rpm, y=mid_curve_912_s_power, name="Mid curve")
    )
    # fig_mid_curve.show()

    # Here we should take a curve that is somewhere between the max output or power required for
    # propeller operation This seems to make it possible to match the value in the POH for the fuel
    # consumption.

    data_rpm_fuel_912_s = np.array([2496.2, 2993.9, 3491.3, 3989.4, 4487.5, 4985.4, 5482.0])
    data_fuel_fuel_912_s = np.array([5.99, 6.90, 8.17, 12.0, 15.9, 20.0, 25.5])  # In l/h

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

    # First establish some regressions between power and rpm ?
    poly_power_912_a = np.polyfit(data_power_power_912_a, data_rpm_power_912_a, 3)
    poly_power_912_s = np.polyfit(data_power_power_912_s, data_rpm_power_912_s, 3)
    poly_power_914 = np.polyfit(data_power_power_914, data_rpm_power_914, 3)

    fig912a = go.Figure()

    fig912a.add_trace(
        go.Scatter(x=data_power_power_912_a, y=data_rpm_power_912_a, name="Original data")
    )
    fig912a.add_trace(
        go.Scatter(
            x=data_power_power_912_a,
            y=np.polyval(poly_power_912_a, data_power_power_912_a),
            name="Interpolated data",
        )
    )
    fig912a.update_layout(title_text="Power/RPM interpolation 912-A")

    # fig912a.show()

    fig912s = go.Figure()

    fig912s.add_trace(
        go.Scatter(x=data_power_power_912_s, y=data_rpm_power_912_s, name="Original data")
    )
    fig912s.add_trace(
        go.Scatter(
            x=data_power_power_912_s,
            y=np.polyval(poly_power_912_s, data_power_power_912_s),
            name="Interpolated data",
        )
    )
    fig912s.update_layout(title_text="Power/RPM interpolation 912-S")

    # fig912s.show()

    fig914 = go.Figure()

    fig914.add_trace(
        go.Scatter(
            x=data_power_power_914,
            y=data_rpm_power_914,
            name="Original data",
            mode="lines+markers",
        )
    )
    fig914.add_trace(
        go.Scatter(
            x=data_power_power_914,
            y=np.polyval(poly_power_914, data_power_power_914),
            name="Interpolated data",
            mode="lines+markers",
        )
    )
    fig914.update_layout(title_text="Power/RPM interpolation 914")

    # fig914.show()

    # We'll focus on max continuous performances, 5500 seems to be a common upper limit for
    # continuous operation
    mcp_914 = 73.5  # kW at 5500 rpm
    mcp_912a = 58.0  # kW at 5500 rpm
    mcp_912s = 69.0  # kW at 5500 rpm

    displacement_914 = 1211  # cm**3
    displacement_912a = 1211  # cm**3
    displacement_912s = 1352  # cm**3

    # Very big difference in the power for the same displacement between 912 and 914 which means
    # different max pressure which should essentially be linked to the injection system. For this
    # model we'll consider constant motor architecture so in the end it will be a 912-like ICE.

    print(
        "MEP max for Rotax 912-A:",
        mean_effective_pressure(
            displacement_912a / 1e6, mcp_912a * 1000.0, 5500 * 2.0 * np.pi / 60.0
        )
        * 1e-5,
    )
    print(
        "MEP max for Rotax 912-S:",
        mean_effective_pressure(
            displacement_912s / 1e6, mcp_912s * 1000.0, 5500 * 2.0 * np.pi / 60.0
        )
        * 1e-5,
    )

    # Around 5% difference in the max MEP...

    print(
        "MEP for 912-A along propeller power curve",
        mean_effective_pressure(
            displacement_912a / 1e6,
            data_power_power_912_a * 1000.0,
            data_rpm_power_912_a * 2.0 * np.pi / 60.0,
        )
        * 1e-5,
    )

    print(
        "MEP for 912-S along propeller power curve",
        mean_effective_pressure(
            displacement_912s / 1e6,
            data_power_power_912_s * 1000.0,
            data_rpm_power_912_s * 2.0 * np.pi / 60.0,
        )
        * 1e-5,
    )

    # In the previous case, sfc was a function of MEP and RPM but in the case of Rotax both are
    # linked because RPM gives you power which gives you torque which gives you MEP. What we need
    # to verify is if there is a either a constant sfc or if the function of SFC based on MEP is
    # similar for both engine

    density_avgas = 0.72  # In kg/l

    poly_rpm_to_power_912_a = np.polyfit(data_rpm_power_912_a, data_power_power_912_a, 3)
    poly_rpm_to_power_912_s = np.polyfit(data_rpm_power_912_s, data_power_power_912_s, 3)

    sfc_fuel_curve_912a = data_fuel_fuel_912_a / np.polyval(
        poly_rpm_to_power_912_a, data_rpm_fuel_912_a
    )  # In l/h/kW
    sfc_fuel_curve_912a *= density_avgas  # In kg/h/W
    sfc_fuel_curve_912a *= 1000  # In g/Wh
    print(sfc_fuel_curve_912a)

    sfc_fuel_curve_912s = data_fuel_fuel_912_s / np.polyval(
        poly_rpm_to_power_912_s, data_rpm_fuel_912_s
    )  # In l/h/kW
    sfc_fuel_curve_912s *= density_avgas  # In kg/h/W
    sfc_fuel_curve_912s *= 1000  # In g/Wh
    print(sfc_fuel_curve_912s)

    fig_fuel = go.Figure()
    fig_fuel.add_trace(
        go.Scatter(
            x=np.polyval(poly_rpm_to_power_912_a, data_rpm_fuel_912_a),
            y=sfc_fuel_curve_912a,
            name="Original data",
            mode="lines+markers",
        )
    )

    # fig_fuel.show()

    mep_for_sfc_912a = (
        mean_effective_pressure(
            displacement_912a / 1e6,
            np.polyval(poly_rpm_to_power_912_a, data_rpm_fuel_912_a) * 1000.0,
            data_rpm_fuel_912_a * 2.0 * np.pi / 60.0,
        )
        * 1e-5
    )
    fig_fuel_mep = go.Figure()
    fig_fuel_mep.add_trace(
        go.Scatter(
            # x=mean_effective_pressure(
            #     displacement_912a / 1e6,
            #     np.polyval(poly_rpm_to_power_912_a, data_rpm_fuel_912_a) * 1000.0,
            #     data_rpm_fuel_912_a * 2.0 * np.pi / 60.0,
            # )
            # * 1e-5,
            x=np.polyval(poly_rpm_to_power_912_a, data_rpm_fuel_912_a),
            y=sfc_fuel_curve_912a,
            name="Data Rotax 912-A",
            mode="lines+markers",
        )
    )
    mep_for_sfc_912s = (
        mean_effective_pressure(
            displacement_912s / 1e6,
            np.polyval(poly_rpm_to_power_912_s, data_rpm_fuel_912_s) * 1000.0,
            data_rpm_fuel_912_s * 2.0 * np.pi / 60.0,
        )
        * 1e-5
    )
    fig_fuel_mep.add_trace(
        go.Scatter(
            # x=mep_for_sfc_912s,
            x=np.polyval(poly_rpm_to_power_912_s, data_rpm_fuel_912_s),
            y=sfc_fuel_curve_912s,
            name="Data Rotax 912-S",
            mode="lines+markers",
        )
    )

    # fig_fuel_mep.show()

    # There seems like there isn't really any constance in the sfc = f(MEP) as what was found for
    # the original model. The shape is more or less the same though. So what we will do is derive
    # a regression based on the min and max sfc whose evolution seems consistent with previous
    # models. We'll take something under the form of a + k * exp(-(x-MEP_MAX) with k chosen so
    # that the value of the sfc is equal to the max value of the sfc for x = Min MEP (which we'll
    # take as 5 bar). For the max value of the MEP we'll take 18.

    # Values obtained from reading the curves plotted before
    sfc_mep_min_912a = np.interp(5, mep_for_sfc_912a, sfc_fuel_curve_912a)
    sfc_mep_min_912s = np.interp(5, mep_for_sfc_912s, sfc_fuel_curve_912s)

    sfc_mep_max_912a = np.interp(18, mep_for_sfc_912a, sfc_fuel_curve_912a)
    sfc_mep_max_912s = np.interp(18, mep_for_sfc_912s, sfc_fuel_curve_912s)

    print(sfc_mep_min_912a, sfc_mep_min_912s)
    print(sfc_mep_max_912a, sfc_mep_max_912s)

    fig_fuel_interp = go.Figure()

    sfc_interp_912a = sfc_mep_max_912a + (sfc_mep_min_912a - sfc_mep_max_912a) / (
        np.exp(18.0 - 5.0)
    ) * np.exp(-(mep_for_sfc_912a - 18.0))

    fig_fuel_interp.add_trace(
        go.Scatter(
            x=mep_for_sfc_912s,
            y=sfc_interp_912a,
            name="Interpolation",
            mode="lines+markers",
        )
    )
    fig_fuel_interp.add_trace(
        go.Scatter(
            x=mep_for_sfc_912s,
            y=sfc_fuel_curve_912a,
            name="Data Rotax 912-S",
            mode="lines+markers",
        )
    )

    # fig_fuel_interp.show()

    fig_fuel_interp_2 = go.Figure()

    k_a = np.log(sfc_mep_min_912a - sfc_mep_max_912a) / (18.0 - 5.0)
    sfc_interp_912a_v2 = sfc_mep_max_912a + np.exp(-k_a * (mep_for_sfc_912a - 18.0))

    fig_fuel_interp_2.add_trace(
        go.Scatter(
            x=mep_for_sfc_912a,
            y=sfc_interp_912a_v2,
            name="Interpolation Rotax 912-A",
            mode="lines+markers",
        )
    )
    fig_fuel_interp_2.add_trace(
        go.Scatter(
            x=mep_for_sfc_912a,
            y=sfc_fuel_curve_912a,
            name="Data Rotax 912-A",
            mode="lines+markers",
        )
    )

    k_s = np.log(sfc_mep_min_912s - sfc_mep_max_912s) / (18.0 - 5.0)
    print("k coefficient for Rotax 912-S", k_s)
    sfc_interp_912s_v2 = sfc_mep_max_912s + np.exp(-k_s * (mep_for_sfc_912s - 18.0))

    fig_fuel_interp_2.add_trace(
        go.Scatter(
            x=mep_for_sfc_912s,
            y=sfc_interp_912s_v2,
            name="Interpolation Rotax 912-S",
            mode="lines+markers",
        )
    )
    fig_fuel_interp_2.add_trace(
        go.Scatter(
            x=mep_for_sfc_912s,
            y=sfc_fuel_curve_912s,
            name="Data Rotax 912-S",
            mode="lines+markers",
        )
    )

    # fig_fuel_interp_2.show()

    r2 = 1.0 - np.sum((sfc_fuel_curve_912a - sfc_interp_912a_v2) ** 2) / np.sum(
        (sfc_fuel_curve_912a - np.mean(sfc_fuel_curve_912a)) ** 2
    )
    print("R2: ", r2)
