import numpy as np
import openmdao.api as om

from stdatm import Atmosphere

import plotly.graph_objects as go

from tests.testing_utilities import run_system
from fastga_he.models.propulsion.components.source.turboshaft.components.perf_fuel_consumption import PerformancesTurboshaftFuelConsumption

if __name__ == "__main__":

    rho_0 = Atmosphere(0).density
    atm = Atmosphere(altitude=np.full(50, 10000), altitude_in_feet=True)
    atm.true_airspeed = 120 * 0.5144
    throttle_ratio = np.linspace(33, 100, 50) / 100.0

    ivc_pw206b = om.IndepVarComp()

    design_power_pw206 = 308  # In kW
    ivc_pw206b.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="kW", val=design_power_pw206
    )
    ivc_pw206b.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:T41t",
        units="degK",
        val=1400.0,
    )
    ivc_pw206b.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:OPR", val=8.0
    )
    ivc_pw206b.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:power_ratio", val=1.56
    )
    ivc_pw206b.add_output("density_ratio", val=atm.density / rho_0)
    ivc_pw206b.add_output("mach", val=atm.mach)
    ivc_pw206b.add_output("power_required", val=throttle_ratio * design_power_pw206, units="kW")
    ivc_pw206b.add_output(
        "settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_sfc", val=1.11
    )

    problem = run_system(
        PerformancesTurboshaftFuelConsumption(turboshaft_id="turboshaft_1", number_of_points=50),
        ivc_pw206b,
    )

    sfc_pw206b = (
        problem.get_val("fuel_consumption", units="kg/h") / problem.get_val("power_required", units="kW")
    )

    ivc_pt6a_34 = om.IndepVarComp()

    design_power_pt6a_34 = 551  # In kW
    ivc_pt6a_34.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="kW", val=design_power_pt6a_34
    )
    ivc_pt6a_34.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:T41t",
        units="degK",
        val=1400.0,
    )
    ivc_pt6a_34.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:OPR", val=7.0
    )
    ivc_pt6a_34.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:power_ratio", val=1.2
    )
    ivc_pt6a_34.add_output("density_ratio", val=atm.density / rho_0)
    ivc_pt6a_34.add_output("mach", val=atm.mach)
    ivc_pt6a_34.add_output("power_required", val=throttle_ratio * design_power_pt6a_34, units="kW")
    ivc_pt6a_34.add_output(
        "settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_sfc", val=1.05
    )

    problem = run_system(
        PerformancesTurboshaftFuelConsumption(turboshaft_id="turboshaft_1", number_of_points=50),
        ivc_pt6a_34,
    )

    sfc_pt6a_34 = (
            problem.get_val("fuel_consumption", units="kg/h") / problem.get_val("power_required", units="kW")
    )

    fig = go.Figure()

    scatter_pw206b = go.Scatter(
        x=throttle_ratio * design_power_pw206,
        y=sfc_pw206b,
        name="Pratt and Whitney PW206B",
        mode="markers",
        marker={
            "symbol": "diamond",
            "color": "red",
            "size": 12
        },
        showlegend=True
    )
    fig.add_trace(scatter_pw206b)

    scatter_pt6a_34 = go.Scatter(
        x=throttle_ratio * design_power_pt6a_34,
        y=sfc_pt6a_34,
        name="Pratt and Whitney PT6A-34",
        mode="markers",
        marker={
            "symbol": "cross",
            "color": "blue",
            "size": 12
        },
        showlegend=True
    )
    fig.add_trace(scatter_pt6a_34)

    fig.update_layout(
        xaxis_title="Shaft power [kW]",
        yaxis_title="PSFC [kg/kW/h]",
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        plot_bgcolor="white",
        title_font=dict(size=20),
    )

    fig.update_xaxes(
        ticks="outside",
        title_font=dict(size=15),
        tickfont=dict(size=15),
        showline=True,
        linecolor="black",
        linewidth=3,
    )

    fig.update_yaxes(
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
    )

    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()