#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

import plotly.graph_objects as go

from fastga_he.gui.lca_impact import (
    lca_score_sensitivity_simple,
    lca_score_sensitivity_advanced_components_and_phase,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_lca"
SENSITIVITY_RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results_sensitivity_2"


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
