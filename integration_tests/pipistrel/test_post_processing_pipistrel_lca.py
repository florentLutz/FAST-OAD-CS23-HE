#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import time
import pathlib

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import fastoad.api as oad

from fastga_he.gui.lca_impact import (
    lca_score_sensitivity_simple,
    lca_score_sensitivity_advanced_components_and_phase,
    lca_impacts_bar_chart_simple,
    lca_impacts_bar_chart_normalised_weighted,
    lca_raw_impact_comparison_advanced,
    lca_impacts_search_table,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_lca"
ORIG_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"
SENSITIVITY_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_sensitivity_2"
SENSITIVITY_CELLS_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_sensitivity_lca"


def test_compare_single_scores_evolution_with_and_without_aging():
    """
    On a single graph, compare the evolution of the single score of the Pipistrel with lifetime
    with and without aging effect.
    """
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_RESULTS_FOLDER_PATH,
        prefix="reference",
        name="Pipistrel without battery aging model",
    )

    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_RESULTS_FOLDER_PATH,
        prefix="full_aging",
        name="Pipistrel with battery aging model",
        fig=fig,
    )

    # We do that so that the legend doesn't overlap the y-axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()


def test_lca_sensitivity_analysis_advanced_impact_categories_and_phase():
    """
    Plots the evolutions of the contribution to the single score of the components and the phase
    they are used in. Uses ReCiPe which might lead to some weird post-processing but the general
    order of magnitude seems OK
    """
    fig = lca_score_sensitivity_advanced_components_and_phase(
        results_folder_path=SENSITIVITY_RESULTS_FOLDER_PATH,
        prefix="full_aging",
        name="Pipistrel with battery aging model",
        cutoff_criteria=3,
        force_order=["airframe", "electricity_for_mission", "battery_pack_1", "battery_pack_2"],
    )

    fig.add_vline(x=4000.0, line_width=3, line_dash="dash", line_color="red")
    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


def test_visualize_mass_divergence_long_lifespan_cell():
    # Data, which are results of the previous test runs for different value of the cell weight.
    battery_energy_density = np.array([130, 120, 110, 100, 90, 80, 75])
    mtow = np.array([862.0, 930, 1017, 1147, 1355, 1796, 2451])
    m_bat = np.array([164.6, 186.93, 216.48, 259.4, 327.75, 468.68, 671.05])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=battery_energy_density,
            y=mtow,
            name="MTOW [kg]",
            yaxis="y1",
            line=dict(color="black", width=3),
            marker={
                "symbol": "diamond",
                "size": 15,
                "color": "black",
            },
        )
    )

    fig.add_trace(
        go.Scatter(
            x=battery_energy_density,
            y=m_bat,
            name="Battery mass [kg]",
            line=dict(color="red", dash="dash", width=3),
            marker={
                "symbol": "circle",
                "size": 15,
                "color": "red",
            },
        )
    )
    fig.add_vline(x=67.9, line_width=1, line_dash="dash", line_color="grey")
    fig.add_annotation(
        x=67.9 + 0.4,
        y=0.0,
        yref="paper",
        text="Battery energy density of the original cell",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=20, color="grey"),
    )

    fig.update_layout(
        # title="Evolution of battery mass and MTOW with battery energy density",
        xaxis=dict(title="Battery energy density [Wh/kg]", range=[60, None]),
        yaxis=dict(title="MTOW, Battery mass [kg]"),
        template="plotly_white",
        font=dict(size=20),
        height=800,
        width=1600,
    )

    fig.update_layout(
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        height=800,
        width=1600,
        margin=dict(l=5, r=5, t=60, b=5),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
        range=[62.9, 135],
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
    )

    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()

    pdf_path = "results/pipistrel_feasibility_battery_energy_density.pdf"

    write = True

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)


def test_compare_impacts_three_designs_simple_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_reference_cell.xml",
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_energy_density_cell.xml",
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_lifespan_cell.xml",
        ],
        names_aircraft=[
            "Pipistrel with reference cell",
            "Pipistrel with high energy density cell",
            "Pipistrel with high lifespan cell",
        ],
    )

    fig.show()


def test_compare_impacts_three_designs_bar_chart_normalised():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_reference_cell.xml",
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_energy_density_cell.xml",
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_lifespan_cell.xml",
        ],
        names_aircraft=[
            "Ref. NMC cell",
            "Si-NMC cell",
            "SuperBattery cell",
        ],
        impact_filter_list=[
            "acidification",
            "climate change",
            "ecotoxicity freshwater",
            "energy resources non-renewable",
            "eutrophication freshwater",
            "eutrophication marine",
            "eutrophication terrestrial",
            "human toxicity carcinogenic",
            "human toxicity non-carcinogenic",
            "ionising radiation human health",
            "land use",
            "material resources metals minerals",
            "ozone depletion",
            "particulate matter formation",
            "photochemical oxidant formation human health",
            "water use",
        ],
    )
    fig.update_layout(
        title=None,
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_font=dict(size=20),
        legend_font=dict(size=20),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        width=1800,
        height=800,
    )
    fig.update_xaxes(
        title_font=dict(size=15),
    )
    fig.update_yaxes(
        title_font=dict(size=15),
    )
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()

    write = False

    if write:
        pdf_path = "results/impacts_evolution_pipistrel_battery.pdf"

        pio.write_image(fig, pdf_path, width=1900, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1900, height=900)


def test_compare_impacts_three_designs_with_contributor():
    fig = lca_raw_impact_comparison_advanced(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_reference_cell.xml",
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_energy_density_cell.xml",
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_lifespan_cell.xml",
        ],
        names_aircraft=[
            "Ref. NMC cell",
            "Si-NMC cell",
            "SuperBattery cell",
        ],
        impact_category="material resources metals minerals",  # "climate change", "material resources metals minerals", "energy resources non-renewable"
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well
            "Battery pack": ["battery_pack_1", "battery_pack_2"],
            "Use phase": "electricity_for_mission",  # Just a renaming, should work as well
            "Others": [
                "propeller_1",
                "motor_1",
                "inverter_1",
                "harness_1",
                "dc_bus_1",
                "manufacturing",
                "distribution",
                "dc_sspc_1",
                "dc_sspc_2",
                "dc_splitter_1",
            ],
        },
    )
    fig.update_layout(width=700, height=900)

    fig.update_layout(
        title=None,
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_font=dict(size=20),
        legend_font=dict(size=20),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    fig.update_xaxes(
        title_font=dict(size=15),
    )
    fig.update_yaxes(
        title_font=dict(size=15),
    )
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()

    write = False

    if write:
        pdf_path = "results/lca_material_resources_minerals_contributors.pdf"

        pio.write_image(fig, pdf_path, width=700, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=700, height=900)


def test_search_engine():
    impact_list = ["*", "*", "*"]
    phase_list = ["production", "*", "operation"]
    component_list = ["*", "battery_pack_1", "*"]

    impacts_value = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_energy_density_cell.xml",
        impact_list,
        phase_list,
        component_list,
        rel=True,
    )

    print(impacts_value[0], impacts_value[1], impacts_value[2])

    impact_list = ["*"]
    phase_list = ["*"]
    component_list = ["battery_pack_1"]

    impacts_value = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_energy_density_cell.xml",
        impact_list,
        phase_list,
        component_list,
        rel=False,
    )
    datafile = oad.DataFile(
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_energy_density_cell.xml"
    )
    batt_mass_per_fu = datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass_per_fu"
    ].value[0]

    print("Battery contribution to single score", impacts_value[0] * 2.0)
    print("Single score per kg of battery", impacts_value[0] / batt_mass_per_fu)


def test_search_engine_high_lifespan():
    impact_list = ["*", "*"]
    phase_list = ["production", "*"]
    component_list = ["*", "battery_pack_1"]

    impacts_value = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_lifespan_cell.xml",
        impact_list,
        phase_list,
        component_list,
        rel=True,
    )

    print(impacts_value[0], impacts_value[1])

    impact_list = ["*"]
    phase_list = ["*"]
    component_list = ["battery_pack_1"]

    impacts_value = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_lifespan_cell.xml",
        impact_list,
        phase_list,
        component_list,
        rel=False,
    )
    datafile = oad.DataFile(RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_high_lifespan_cell.xml")
    batt_mass_per_fu = datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass_per_fu"
    ].value[0]

    print("Battery contribution to single score", impacts_value[0] * 2.0)
    print("Single score per kg of battery", impacts_value[0] / batt_mass_per_fu)


def test_search_engine_default_cell():
    impact_list = ["*"]
    phase_list = ["*"]
    component_list = ["battery_pack_1"]

    impacts_value = lca_impacts_search_table(
        RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_reference_cell.xml",
        impact_list,
        phase_list,
        component_list,
        rel=False,
    )
    datafile = oad.DataFile(RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_reference_cell.xml")
    batt_mass_per_fu = datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass_per_fu"
    ].value[0]

    print("Battery contribution to single score", impacts_value[0] * 2.0)
    print("Single score per kg of battery", impacts_value[0] / batt_mass_per_fu)


def test_compare_sensitivity_three_designs():
    """
    On a single graph, compare the evolution of the single score of the Pipistrel sized with 3
    different cells
    """
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_CELLS_RESULTS_FOLDER_PATH,
        prefix="reference_cell",
        name="Pipistrel with reference cell",
    )

    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_CELLS_RESULTS_FOLDER_PATH,
        prefix="high_energy_density_cell",
        name="Pipistrel with high energy density cell",
        fig=fig,
    )

    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_CELLS_RESULTS_FOLDER_PATH,
        prefix="high_lifespan_cell",
        name="Pipistrel with high lifespan cell",
        fig=fig,
    )

    # We do that so that the legend doesn't overlap the y-axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()


def test_lca_sensitivity_analysis_high_lifespan_cell():
    fig = lca_score_sensitivity_advanced_components_and_phase(
        results_folder_path=SENSITIVITY_CELLS_RESULTS_FOLDER_PATH,
        prefix="high_lifespan_cell",
        name="Pipistrel with high lifespan cell",
        cutoff_criteria=3,
        force_order=["airframe", "electricity_for_mission", "battery_pack_1", "battery_pack_2"],
    )

    fig.add_vline(x=4000.0, line_width=3, line_dash="dash", line_color="red")
    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


def test_compare_impacts_with_and_without_aging():
    fig = lca_impacts_bar_chart_simple(
        [
            ORIG_RESULTS_FOLDER_PATH / "pipistrel_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca_proper_aging_with_econ.xml",
        ],
        names_aircraft=[
            "Pipistrel without aging consideration",
            "Pipistrel with aging consideration",
        ],
    )

    fig.update_layout(
        title=None,
        showlegend=True,
        margin=dict(l=5, r=5, t=60, b=5),
        title_font=dict(size=20),
        legend_font=dict(size=20),
    )
    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()
    write = True

    if write:
        pdf_path = "results/effect_aging_on_lca.pdf"

        pio.write_image(fig, pdf_path, width=1900, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1900, height=900)
