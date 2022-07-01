# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import scipy.optimize as opt

import plotly.graph_objects as go

from fastga_he.models.propulsion.components.propulsor.propeller.components.sizing_propeller_section_aero import (
    SizingPropellerSectionAero,
    SizingPropellerSectionAeroIdentification,
)


def generic_twist_law(radius_distribution, amplitude_parameter, scaling_parameter, k2, k1, k0):

    return (
        scaling_parameter * np.arctan2(amplitude_parameter, radius_distribution)
        + k2 * radius_distribution ** 2.0
        + k1 * radius_distribution
        + k0
    )


def alpha_l_d_max():
    problem = om.Problem()
    model = problem.model
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:diameter", val=2.0, units="m"
    )
    ivc.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:elements_radius",
        val=[0.0, 0.25, 0.28, 0.35, 0.40, 0.45],
        units="m",
    )
    model.add_subsystem(
        name="inputs",
        subsys=ivc,
        promotes=["*"],
    )
    model.add_subsystem(
        name="aerodynamic_coeff",
        subsys=SizingPropellerSectionAero(propeller_id="propeller_1", elements_number=6),
        promotes=["*"],
    )

    problem.setup()
    problem.run_model()

    alpha = problem["data:propulsion:he_power_train:propeller:propeller_1:alpha_list"]
    cl = problem["data:propulsion:he_power_train:propeller:propeller_1:cl_array"]
    cd = problem["data:propulsion:he_power_train:propeller:propeller_1:cd_array"]

    alpha_max_l_d_list = []

    for idx, value in enumerate(cl):
        alpha_loc, cl_loc, cd_loc = SizingPropellerSectionAeroIdentification.reshape_polar(
            alpha, value, cd[idx]
        )
        alpha_max_l_d = alpha[np.argmax(cl_loc / cd_loc)]
        alpha_max_l_d_list.append(alpha_max_l_d)

    return alpha_max_l_d_list


if __name__ == "__main__":

    radius_tbm900 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]) * 2.31 / 2.0
    twist_tbm900 = np.array([20.0, 15.5, 10.0, 5.0, 1.0, -2.0, -4.0, -5.0, -5.5]) * np.pi / 180.0

    bnds = ((-np.inf, 0.0, -10, -10, -10), (np.inf, np.pi / 2, 10, 10, 10))
    x0 = (1.0, 1.0, 1.0, 1.0, 0.0)

    p_opt, p_cov = opt.curve_fit(generic_twist_law, radius_tbm900, twist_tbm900, x0, bounds=bnds)
    fit_coeff_1, fit_coeff_2, fit_coeff_3, fit_coeff_4, fit_coeff_5 = p_opt

    fig = go.Figure()
    orig_twist_tbm900 = go.Scatter(
        x=radius_tbm900,
        y=twist_tbm900,
        mode="lines+markers",
        name="Twist law TBM900 propeller in FAST-OAD-CS23",
        legendgroup="TBM900",
        legendgrouptitle_text="TBM900",
    )
    fig.add_trace(orig_twist_tbm900)

    fitted_twist_tbm900 = go.Scatter(
        x=radius_tbm900,
        y=generic_twist_law(
            radius_tbm900,
            fit_coeff_1,
            fit_coeff_2,
            fit_coeff_3,
            fit_coeff_4,
            fit_coeff_5,
        ),
        mode="lines+markers",
        name="Twist law TBM900 propeller with 2 parameters distribution",
        legendgroup="TBM900",
        legendgrouptitle_text="TBM900",
    )
    fig.add_trace(fitted_twist_tbm900)

    fitted_twist_tbm900_diff = go.Scatter(
        x=radius_tbm900,
        y=generic_twist_law(
            radius_tbm900,
            fit_coeff_1,
            fit_coeff_2,
            fit_coeff_3,
            fit_coeff_4,
            fit_coeff_5,
        )
        - twist_tbm900,
        mode="lines+markers",
        name="Difference in twist law TBM900 propeller with 2 parameters distribution",
    )
    fig.add_trace(fitted_twist_tbm900_diff)

    fig.show()
