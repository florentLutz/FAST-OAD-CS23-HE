#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

import numpy as np
import plotly.graph_objects as go

from fastga_he.gui.lca_impact import (
    lca_score_sensitivity_simple,
    lca_score_sensitivity_advanced_components_and_phase,
    lca_impacts_bar_chart_simple,
    lca_impacts_bar_chart_normalised,
    lca_raw_impact_comparison_advanced,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_lca"
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
    battery_energy_density = np.array([129, 115, 103, 98.5, 97.6, 96.7])
    mtow = np.array([961.0, 1123, 1388, 1871, 1960, 2463])
    m_bat = np.array([195, 248, 333, 477, 506, 661])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=battery_energy_density, y=mtow, name="MTOW [kg]", yaxis="y1", line=dict(color="blue")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=battery_energy_density,
            y=m_bat,
            name="Battery mass [kg]",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.add_vline(x=65.0, line_width=3, line_dash="dash", line_color="black")

    fig.update_layout(
        title="Evolution of battery mass and MTOW with battery energy density",
        xaxis=dict(title="Battery energy density [W*h/kg]", range=[60, None]),
        yaxis=dict(title="MTOW [kg] / Battery mass [kg]"),
        template="plotly_white",
        title_x=0.5,
        font=dict(size=15),
        height=800,
        width=1600,
    )

    fig.show()


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
    fig = lca_impacts_bar_chart_normalised(
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


def test_compare_impacts_three_designs_with_contributor():
    fig = lca_raw_impact_comparison_advanced(
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
        impact_category="ionising radiation human health",  # "climate change", "material resources metals minerals"
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
    fig.update_layout(width=800)

    fig.show()


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
