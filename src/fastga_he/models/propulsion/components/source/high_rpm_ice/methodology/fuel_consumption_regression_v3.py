# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go

from stdatm import Atmosphere


def r2(ref_values, interp_values):
    return 1.0 - np.sum((ref_values - interp_values) ** 2) / np.sum(
        (ref_values - np.mean(ref_values)) ** 2
    )


if __name__ == "__main__":
    sfc_f_rpm_rpm_912s = np.array(
        [
            5482.566647579055,
            4984.125469237132,
            4489.215499141058,
            3983.043837882548,
            3483.775529681237,
            2984.9207864096206,
        ]
    )  # In rpm
    sfc_f_rpm_k_rpm_912s = sfc_f_rpm_rpm_912s / sfc_f_rpm_rpm_912s[0]
    sfc_f_rpm_bsfc_912s = np.array(
        [
            286.21874403512135,
            290.09671056817473,
            296.12680536998164,
            304.75281542279066,
            319.8383915505504,
            329.32016288095696,
        ]
    )  # In g/kWh
    sfc_f_rpm_k_sfc = sfc_f_rpm_bsfc_912s / sfc_f_rpm_bsfc_912s[0]

    sfc_f_alt_alt_912s = np.array(
        [
            66.83303439132851,
            1822.896358106524,
            3318.0223036730868,
            4853.076503158388,
            6468.923028932389,
            8187.241675115029,
            10006.160804803869,
            11886.99992201513,
            13950.479606956253,
            16096.623255088512,
            18202.526709818296,
        ]
    )  # In feet

    # Density ratio
    sfc_f_alt_sigma = (
        Atmosphere(sfc_f_alt_alt_912s, altitude_in_feet=True).density / Atmosphere(0).density
    )

    sfc_f_alt_bsfc_912s = np.array(
        [
            289.5352101692272,
            303.3525696014973,
            318.7405443343991,
            333.35023005536925,
            348.7288466037588,
            368.75068236762064,
            388.76471964438895,
            412.6499259143725,
            440.39694299305927,
            473.56390860173116,
            506.73399360524047,
        ]
    )  # In g/kWh
    sfc_f_alt_k_sfc = sfc_f_alt_bsfc_912s / sfc_f_alt_bsfc_912s[0]

    # RPM effect
    f_rpm_rpm_interp = np.linspace(0.3, 1.0)
    poly_sfc_f_rpm = np.polyfit(sfc_f_rpm_k_rpm_912s, sfc_f_rpm_k_sfc, 3)

    fig_rpm_effect = go.Figure()
    fig_rpm_effect.add_trace(
        go.Scatter(
            x=sfc_f_rpm_k_rpm_912s,
            y=sfc_f_rpm_k_sfc,
            name="Data Rotax 912-S",
            mode="lines+markers",
        )
    )
    fig_rpm_effect.add_trace(
        go.Scatter(
            x=f_rpm_rpm_interp,
            y=np.polyval(poly_sfc_f_rpm, f_rpm_rpm_interp),
            name="Interpolation Rotax 912-S",
            mode="lines+markers",
        )
    )
    fig_rpm_effect.update_layout(
        title_text="RPM effect on sfc",
        title_x=0.5,
        xaxis_title="RPM to max RPM",
        yaxis_title="SFC to SFC @ max RPM",
    )
    # fig_rpm_effect.show()
    print("Polynomial for RPM effect:", poly_sfc_f_rpm)

    # Altitude effect
    f_alt_sigma_interp = np.linspace(1.0, 0.3)
    poly_sfc_f_alt = np.polyfit(sfc_f_alt_sigma, sfc_f_alt_k_sfc, 2)

    fig_alt_effect = go.Figure()
    fig_alt_effect.add_trace(
        go.Scatter(
            x=sfc_f_alt_sigma,
            y=sfc_f_alt_k_sfc,
            name="Data Rotax 912-S",
            mode="lines+markers",
        )
    )
    fig_alt_effect.add_trace(
        go.Scatter(
            x=f_alt_sigma_interp,
            y=np.polyval(poly_sfc_f_alt, f_alt_sigma_interp),
            name="Interpolation Rotax 912-S",
            mode="lines+markers",
        )
    )
    fig_alt_effect.update_layout(
        title_text="Altitude effect on sfc",
        title_x=0.5,
        xaxis_title="Density ratio",
        yaxis_title="SFC to SFC @SL",
    )
    # fig_alt_effect.show()
    print("Polynomial for altitude effect:", poly_sfc_f_alt)

    # Testing the regression at the validation point (cruise for the Pipistrel SW121)
    sigma_validation_point = (
        Atmosphere(2000.0, altitude_in_feet=True).density / Atmosphere(0).density
    )
    rpm_ratio_validation_point = 5300.0 / 5500.0

    # Base value times the two effects
    sfc_validation_point = (
        285.0
        * np.polyval(poly_sfc_f_rpm, rpm_ratio_validation_point)
        * np.polyval(poly_sfc_f_alt, sigma_validation_point)
    )  # In g/kWh

    # Should be 75% of MCP so
    power_validation_point = 0.75 * 69.0  # In kW
    ff_validation_point = power_validation_point * sfc_validation_point / 1000.0  # In kg/h

    ff_validation_point_volume = ff_validation_point / 0.72

    # Still doesn't give good results, although it is better now that we take into account the
    # altitude effect went from 25.0 l/h to 21.6 lh. A reason that might explain this difference is
    # the fact that these data are obtained with a fixed pitch propeller, while on the SW121 there
    # is a variable pitch propeller.

    print("Fuel flow validation point [l/h]:", ff_validation_point_volume)
    print("Error to ref fuel flow [%]:", (ff_validation_point_volume - 18.4) / 18.4 * 100.0)

    # Let's see if this margin is consistent with other points, in which case we could reasonably
    # account for that unknown effect in a coefficient

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
    )  # In l/h
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

    sigma_validation_points = (
        Atmosphere(altitude_validation_points, altitude_in_feet=True).density
        / Atmosphere(0).density
    )
    rpm_ratio_validation_points = rpm_validation_points / 5500.0

    sfc_validation_points = (
        285.0
        # * np.polyval(poly_sfc_f_rpm, rpm_ratio_validation_points)
        * np.polyval(poly_sfc_f_alt, sigma_validation_points)
    )  # In g/kWh

    # The rpm effect seems to worsen the R2, so it is commented for now

    # Density of 0.72 kg/l
    calculated_vol_ff_validation_point = (
        power_validation_points * sfc_validation_points / 1000.0 / 0.72
    )  # In l/h

    np.set_printoptions(suppress=True)

    print(
        "Error to ref fuel flow validation points [%]:",
        (calculated_vol_ff_validation_point - vol_ff_validation_points)
        / vol_ff_validation_points
        * 100.0,
    )

    print("R2", r2(vol_ff_validation_points, calculated_vol_ff_validation_point))

    # It is not consistent ... Though it seems like the further from the max power at current alt
    # we are, the worse off the difference is. We could try to derive a relation with that but we'd
    # need the data from the SW121, so we would need to validate it somewhere else.

    # For max power at current altitude, we'll use the classical Gagg and Ferrar model.
    p_max_at_altitude_validation_points = (
        sigma_validation_points - (1 - sigma_validation_points) / 7.55
    ) * 69.0

    power_ratio_validation_points = power_validation_points / p_max_at_altitude_validation_points
    print("Power to max power at altitude:", power_ratio_validation_points)

    fig_p_rate_effect = go.Figure()
    fig_p_rate_effect.add_trace(
        go.Scatter(
            x=power_ratio_validation_points,
            y=calculated_vol_ff_validation_point / vol_ff_validation_points,
            mode="markers",
        )
    )
    fig_p_rate_effect.update_layout(
        title_text="Power effect on SFC",
        title_x=0.5,
        xaxis_title="Power to max power at altitude [-]",
        yaxis_title="Error between predicted value and actual value [-]",
    )
    # fig_p_rate_effect.show()

    fig_map_effect = go.Figure()
    fig_map_effect.add_trace(
        go.Scatter(
            x=map_validation_points,
            y=calculated_vol_ff_validation_point / vol_ff_validation_points,
            mode="markers",
        )
    )
    fig_map_effect.update_layout(
        title_text="MAP effect on SFC",
        title_x=0.5,
        xaxis_title="MAP [inHg]",
        yaxis_title="Error between predicted value and actual value [-]",
    )
    # fig_map_effect.show()

    # Altitude effect
    f_alt_sigma_interp_v2 = np.linspace(1.0, 0.3)
    k_sfc_validation_points = (
        vol_ff_validation_points * 0.72 / power_validation_points * 1000.0 / 285.0
    )
    poly_sfc_f_alt_v2 = np.polyfit(sigma_validation_points, k_sfc_validation_points, 2)

    fig_alt_effect_v2 = go.Figure()
    fig_alt_effect_v2.add_trace(
        go.Scatter(
            x=sigma_validation_points,
            y=k_sfc_validation_points,
            name="Data Rotax 912-S",
            mode="markers",
        )
    )
    fig_alt_effect_v2.add_trace(
        go.Scatter(
            x=f_alt_sigma_interp_v2,
            y=np.polyval(poly_sfc_f_alt_v2, f_alt_sigma_interp_v2),
            name="Interpolation Rotax 912-S",
            mode="lines+markers",
        )
    )
    fig_alt_effect_v2.update_layout(
        title_text="Altitude effect on sfc",
        title_x=0.5,
        xaxis_title="Density ratio",
        yaxis_title="SFC to SFC @SL @max rpm",
    )
    fig_alt_effect_v2.show()

    # It seems like there is still a dependency on rpm, not only on altitude, although a second
    # order fit on altitude seems to be decent fit. Time to use PyVPLM I guess ...
