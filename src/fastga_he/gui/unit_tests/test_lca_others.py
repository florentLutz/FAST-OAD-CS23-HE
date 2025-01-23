# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import pathlib

import pytest

import plotly.graph_objects as go

from ..lca_impact import lca_score_sensitivity_simple

PATH_TO_CURRENT_FILE = pathlib.Path(__file__)

SENSITIVITY_STUDIES_FOLDER_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "models"
    / "environmental_impacts"
    / "unit_tests"
    / "results"
    / "parametric_study"
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_score_sensitivity_analysis():
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
    )

    # We do that so that the legend doesn't overlap the y axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_score_sensitivity_analysis_two_plots():
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
    )

    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="ref_kodiak_op",
        name="Reference Kodiak",
        fig=fig
    )

    # We do that so that the legend doesn't overlap the y-axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()

    fig.update_layout(
        height=800.0,
        width=1600.0
    )
    fig.write_image(PATH_TO_CURRENT_FILE.parent / "results" / "evolution_sing_score_kodiak.svg")