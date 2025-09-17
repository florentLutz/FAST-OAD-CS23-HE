import time
import numpy as np
from scipy.interpolate import CubicSpline
import openmdao.api as om

from stdatm import Atmosphere

import plotly.graph_objects as go
import plotly.io as pio

from tests.testing_utilities import run_system
from fastga_he.models.propulsion.components.source.turboshaft.components.perf_fuel_consumption import (
    PerformancesTurboshaftFuelConsumption,
)

if __name__ == "__main__":
    rho_0 = Atmosphere(0).density
    throttle_ratio = np.linspace(33, 100, 50) / 100.0

    ivc_pw206b = om.IndepVarComp()

    design_power_pw206 = 321  # In kW
    power_pw206 = np.arange(150, design_power_pw206, 10)

    atm_pw206 = Atmosphere(altitude=np.full_like(power_pw206, 10000), altitude_in_feet=True)
    atm_pw206.true_airspeed = 120 * 0.5144
    ivc_pw206b.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating",
        units="kW",
        val=design_power_pw206,
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
    ivc_pw206b.add_output("density_ratio", val=atm_pw206.density / rho_0)
    ivc_pw206b.add_output("mach", val=atm_pw206.mach)
    ivc_pw206b.add_output("power_required", val=power_pw206, units="kW")
    ivc_pw206b.add_output(
        "settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_sfc", val=1.11
    )

    problem = run_system(
        PerformancesTurboshaftFuelConsumption(
            turboshaft_id="turboshaft_1", number_of_points=len(power_pw206)
        ),
        ivc_pw206b,
    )

    sfc_pw206b = problem.get_val("fuel_consumption", units="kg/h") / problem.get_val(
        "power_required", units="kW"
    )
    spline_pw206b = CubicSpline(power_pw206, sfc_pw206b)

    ivc_pt6a_34 = om.IndepVarComp()

    design_power_pt6a_34 = 551  # In kW
    power_pt6a34 = np.arange(150, design_power_pt6a_34, 10)
    atm_pt6a34 = Atmosphere(altitude=np.full_like(power_pt6a34, 10000), altitude_in_feet=True)
    atm_pt6a34.true_airspeed = 120 * 0.5144
    ivc_pt6a_34.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating",
        units="kW",
        val=design_power_pt6a_34,
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
    ivc_pt6a_34.add_output("density_ratio", val=atm_pt6a34.density / rho_0)
    ivc_pt6a_34.add_output("mach", val=atm_pt6a34.mach)
    ivc_pt6a_34.add_output("power_required", val=power_pt6a34, units="kW")
    ivc_pt6a_34.add_output(
        "settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_sfc", val=1.05
    )

    problem = run_system(
        PerformancesTurboshaftFuelConsumption(
            turboshaft_id="turboshaft_1", number_of_points=len(power_pt6a34)
        ),
        ivc_pt6a_34,
    )

    sfc_pt6a_34 = problem.get_val("fuel_consumption", units="kg/h") / problem.get_val(
        "power_required", units="kW"
    )
    spline_pt6a_34 = CubicSpline(power_pt6a34, sfc_pt6a_34)

    fig = go.Figure()

    # For cleanliness, we drop a few points of the graph
    values_to_drop_pt6a34 = [
        180,
        190,
        200,
        420,
        430,
        440,
        450,
        460,
        470,
        480,
        490,
        500,
        510,
        520,
        530,
    ]
    index_to_drop_pt6a34 = []
    for value in values_to_drop_pt6a34:
        index_to_drop_pt6a34.append(np.argwhere(power_pt6a34 == value)[0][0])

    power_pt6a34 = np.delete(power_pt6a34, index_to_drop_pt6a34)
    sfc_pt6a_34 = np.delete(sfc_pt6a_34, index_to_drop_pt6a34)

    values_to_drop_pw206 = [210, 200]
    index_to_drop_pw206 = []
    for value in values_to_drop_pw206:
        index_to_drop_pw206.append(np.argwhere(power_pw206 == value)[0][0])

    power_pw206 = np.delete(power_pw206, index_to_drop_pw206)
    sfc_pw206b = np.delete(sfc_pw206b, index_to_drop_pw206)

    scatter_pw206b = go.Scatter(
        # x=throttle_ratio * design_power_pw206,
        # x=throttle_ratio,
        x=power_pw206,
        y=sfc_pw206b,
        name="Hybrid design: Pratt and Whitney PW206B",
        mode="markers",
        marker={"symbol": "triangle-up", "color": "red", "size": 10},
        showlegend=True,
    )

    scatter_pt6a_34 = go.Scatter(
        # x=throttle_ratio * design_power_pt6a_34,
        # x=throttle_ratio,
        x=power_pt6a34,
        y=sfc_pt6a_34,
        name="Original design: Pratt and Whitney PT6A-34",
        mode="markers",
        marker={"symbol": "triangle-down", "color": "black", "size": 10},
        showlegend=True,
    )

    fig.add_trace(scatter_pw206b)
    fig.add_trace(scatter_pt6a_34)
    annotation_x_offset = 2
    annotation_y_offset = 0.002

    cruise_power_orig_design = [186, 201]
    start_scatter_cruise_power_orig_design = go.Scatter(
        x=[cruise_power_orig_design[0]],
        y=[spline_pt6a_34(cruise_power_orig_design[0])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-right", "color": "black", "size": 15},
    )
    fig.add_trace(start_scatter_cruise_power_orig_design)
    end_scatter_cruise_power_orig_design = go.Scatter(
        x=[cruise_power_orig_design[-1]],
        y=[spline_pt6a_34(cruise_power_orig_design[-1])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-left", "color": "black", "size": 15},
    )
    fig.add_trace(end_scatter_cruise_power_orig_design)
    variation_cruise_power_orig_design = np.linspace(
        cruise_power_orig_design[0], cruise_power_orig_design[-1], 50
    )
    variation_scatter_cruise_power_orig_design = go.Scatter(
        x=variation_cruise_power_orig_design,
        y=spline_pt6a_34(variation_cruise_power_orig_design),
        mode="lines",
        showlegend=False,
        line={"color": "black", "width": 3},
    )
    fig.add_trace(variation_scatter_cruise_power_orig_design)
    fig.add_annotation(
        x=np.mean(np.array(cruise_power_orig_design)) + 3.0 * annotation_x_offset,
        y=spline_pt6a_34(np.mean(np.array(cruise_power_orig_design))) - annotation_y_offset,
        text="Cruise original design",
        showarrow=False,
        font=dict(size=20, color="black"),
        xanchor="left",
        yanchor="bottom",
    )

    reserve_power_orig_design = [177, 180]
    start_scatter_reserve_power_orig_design = go.Scatter(
        x=[reserve_power_orig_design[0]],
        y=[spline_pt6a_34(reserve_power_orig_design[0])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-right", "color": "black", "size": 15},
    )
    fig.add_trace(start_scatter_reserve_power_orig_design)
    end_scatter_reserve_power_orig_design = go.Scatter(
        x=[reserve_power_orig_design[-1]],
        y=[spline_pt6a_34(reserve_power_orig_design[-1])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-left", "color": "black", "size": 15},
    )
    fig.add_trace(end_scatter_reserve_power_orig_design)
    variation_reserve_power_orig_design = np.linspace(
        reserve_power_orig_design[0], reserve_power_orig_design[-1], 50
    )
    variation_scatter_reserve_power_orig_design = go.Scatter(
        x=variation_reserve_power_orig_design,
        y=spline_pt6a_34(variation_reserve_power_orig_design),
        mode="lines",
        showlegend=False,
        line={"color": "black", "width": 3},
    )
    fig.add_trace(variation_scatter_reserve_power_orig_design)
    fig.add_annotation(
        x=np.mean(np.array(reserve_power_orig_design)) + annotation_x_offset,
        y=spline_pt6a_34(np.mean(np.array(reserve_power_orig_design))) + annotation_y_offset,
        text="Reserve original design",
        showarrow=False,
        font=dict(size=20, color="black"),
        xanchor="left",
        yanchor="bottom",
    )

    climb_power_orig_design = [420, 525]
    start_scatter_climb_power_orig_design = go.Scatter(
        x=[climb_power_orig_design[0]],
        y=[spline_pt6a_34(climb_power_orig_design[0])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-right", "color": "black", "size": 15},
    )
    fig.add_trace(start_scatter_climb_power_orig_design)
    end_scatter_climb_power_orig_design = go.Scatter(
        x=[climb_power_orig_design[-1]],
        y=[spline_pt6a_34(climb_power_orig_design[-1])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-left", "color": "black", "size": 15},
    )
    fig.add_trace(end_scatter_climb_power_orig_design)
    variation_climb_power_orig_design = np.linspace(
        climb_power_orig_design[0], climb_power_orig_design[-1], 50
    )
    variation_scatter_climb_power_orig_design = go.Scatter(
        x=variation_climb_power_orig_design,
        y=spline_pt6a_34(variation_climb_power_orig_design),
        mode="lines",
        showlegend=False,
        line={"color": "black", "width": 3},
    )
    fig.add_trace(variation_scatter_climb_power_orig_design)
    fig.add_annotation(
        x=np.mean(np.array(climb_power_orig_design)) + annotation_x_offset,
        y=spline_pt6a_34(np.mean(np.array(climb_power_orig_design))) + annotation_y_offset,
        text="Climb original design",
        showarrow=False,
        font=dict(size=20, color="black"),
        xanchor="left",
        yanchor="bottom",
    )
    fig.add_hline(
        y=0.482,
        line_width=1,
        line_dash="dash",
        line_color="black",
        annotation_text="Time average PSFC original design",
        annotation_position="bottom right",
        annotation={"font": {"size": 17, "color": "black"}, "align": "right"},
    )

    cruise_power_h_design = [205, 210]
    start_scatter_cruise_power_h_design = go.Scatter(
        x=[cruise_power_h_design[0]],
        y=[spline_pw206b(cruise_power_h_design[0])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-right", "color": "red", "size": 15},
    )
    fig.add_trace(start_scatter_cruise_power_h_design)
    end_scatter_cruise_power_h_design = go.Scatter(
        x=[cruise_power_h_design[-1]],
        y=[spline_pw206b(cruise_power_h_design[-1])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-left", "color": "red", "size": 15},
    )
    fig.add_trace(end_scatter_cruise_power_h_design)
    variation_cruise_power_h_design = np.linspace(
        cruise_power_h_design[0], cruise_power_h_design[-1], 50
    )
    variation_scatter_cruise_power_h_design = go.Scatter(
        x=variation_cruise_power_h_design,
        y=spline_pw206b(variation_cruise_power_h_design),
        mode="lines",
        showlegend=False,
        line={"color": "red", "width": 3},
    )
    fig.add_trace(variation_scatter_cruise_power_h_design)
    fig.add_annotation(
        x=np.mean(np.array(cruise_power_h_design)) + 3.0 * annotation_x_offset,
        y=spline_pw206b(np.mean(np.array(cruise_power_h_design))) - annotation_y_offset,
        text="Cruise hybrid design",
        showarrow=False,
        font=dict(size=20, color="red"),
        xanchor="left",
        yanchor="bottom",
    )

    reserve_power_h_design = [196, 198]
    start_scatter_reserve_power_h_design = go.Scatter(
        x=[reserve_power_h_design[0]],
        y=[spline_pw206b(reserve_power_h_design[0])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-right", "color": "red", "size": 15},
    )
    fig.add_trace(start_scatter_reserve_power_h_design)
    end_scatter_reserve_power_h_design = go.Scatter(
        x=[reserve_power_h_design[-1]],
        y=[spline_pw206b(reserve_power_h_design[-1])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-left", "color": "red", "size": 15},
    )
    fig.add_trace(end_scatter_reserve_power_h_design)
    variation_reserve_power_h_design = np.linspace(
        reserve_power_h_design[0], reserve_power_h_design[-1], 50
    )
    variation_scatter_reserve_power_h_design = go.Scatter(
        x=variation_reserve_power_h_design,
        y=spline_pw206b(variation_reserve_power_h_design),
        mode="lines",
        showlegend=False,
        line={"color": "red", "width": 3},
    )
    fig.add_trace(variation_scatter_reserve_power_h_design)
    fig.add_annotation(
        x=np.mean(np.array(reserve_power_h_design)) + annotation_x_offset,
        y=spline_pw206b(np.mean(np.array(reserve_power_h_design))) + annotation_y_offset,
        text="Reserve hybrid design",
        showarrow=False,
        font=dict(size=20, color="red"),
        xanchor="left",
        yanchor="bottom",
    )

    climb_power_h_design = [326, 326]
    start_scatter_climb_power_h_design = go.Scatter(
        x=[climb_power_h_design[0]],
        y=[spline_pw206b(climb_power_h_design[0])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-right", "color": "red", "size": 15},
    )
    fig.add_trace(start_scatter_climb_power_h_design)
    end_scatter_climb_power_h_design = go.Scatter(
        x=[climb_power_h_design[-1]],
        y=[spline_pw206b(climb_power_h_design[-1])],
        mode="markers",
        showlegend=False,
        marker={"symbol": "arrow-bar-left", "color": "red", "size": 15},
    )
    fig.add_trace(end_scatter_climb_power_h_design)
    fig.add_annotation(
        x=np.mean(np.array(climb_power_h_design)) + annotation_x_offset,
        y=spline_pw206b(np.mean(np.array(climb_power_h_design))) + annotation_y_offset,
        text="Climb hybrid design",
        showarrow=False,
        font=dict(size=20, color="red"),
        xanchor="left",
        yanchor="bottom",
    )
    fig.add_hline(
        y=0.384,
        line_width=1,
        line_dash="dash",
        line_color="red",
        annotation_text="Time average PSFC hybrid design",
        annotation_position="bottom right",
        annotation={"font": {"size": 17, "color": "red"}, "align": "right"},
    )

    fig.update_layout(
        xaxis_title="Shaft power [kW]",
        yaxis_title="PSFC [kg/kW/h]",
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
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

    fig.show(width=1600, height=900)

    pdf_path = "results/sfc_profile_hybrid_kodiak.pdf"

    write = True

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)
