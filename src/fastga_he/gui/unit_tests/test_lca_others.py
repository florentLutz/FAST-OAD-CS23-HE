# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import pathlib

import pytest

import plotly.graph_objects as go

from fastga_he.exceptions import ImpactUnavailableForPlotError
from ..lca_impact import (
    lca_score_sensitivity_simple,
    lca_score_sensitivity_advanced_impact_category,
    lca_score_sensitivity_advanced_components,
    lca_score_sensitivity_advanced_components_and_phase,
    lca_impacts_bar_chart_simple,
    lca_impacts_bar_chart_normalised_weighted,
    lca_impacts_bar_chart_with_contributors,
    lca_impacts_bar_chart_with_phases_absolute,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULT_FOLDER_PATH = pathlib.Path(__file__).parent / "results"

PATH_TO_CURRENT_FILE = pathlib.Path(__file__)

SENSITIVITY_STUDIES_FOLDER_PATH = (
    pathlib.Path(__file__).parents[2]
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
def test_lca_single_score_sensitivity_analysis_two_plots():
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
        fig=fig,
    )

    # We do that so that the legend doesn't overlap the y-axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()

    fig.update_layout(height=800.0, width=1600.0)
    fig.write_image(PATH_TO_CURRENT_FILE.parent / "results" / "ga_single_score_evolution.pdf")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_other_impact_sensitivity_analysis():
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        impact_to_plot="material_resources_metals_minerals",
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_unavailable_impact_error():
    try:
        _ = lca_score_sensitivity_simple(
            results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
            impact_to_plot="orange_eutrophication",
            prefix="hybrid_kodiak",
        )
    except ImpactUnavailableForPlotError as e:
        assert " unavailable in the output file. Available impacts include: " in e.args[0]


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sensitivity_analysis_advanced_impact_categories():
    fig = lca_score_sensitivity_advanced_impact_category(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
        cutoff_criteria=5,
    )

    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sensitivity_analysis_advanced_components():
    fig = lca_score_sensitivity_advanced_components(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
        cutoff_criteria=2,
    )

    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sensitivity_analysis_advanced_components_and_phase():
    fig = lca_score_sensitivity_advanced_components_and_phase(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
        cutoff_criteria=2,
    )

    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


def test_lca_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "kodiak_100_ef.xml",
            DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml",
        ],
        names_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )

    fig.show()


def test_lca_bar_chart_relative_paper():
    fig = lca_impacts_bar_chart_simple(
        [
            SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml",
            SENSITIVITY_STUDIES_FOLDER_PATH / "hybrid_kodiak_7077.xml",
        ],
        names_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )

    fig.show()
    fig.update_layout(height=800, width=1600)
    fig.write_image(RESULT_FOLDER_PATH / "ga_impacts_evolution.pdf")


def test_lca_bar_chart_normalised_and_weighted():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml"],
        names_aircraft=["Hybrid Kodiak 100"],
    )

    fig.show()


def test_lca_bar_chart_normalised_and_weighted_multiples():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [
            DATA_FOLDER_PATH / "kodiak_100_ef.xml",
            DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml",
        ],
        names_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )

    fig.show()


def test_lca_bar_chart_relative_contribution():
    fig = lca_impacts_bar_chart_with_contributors(
        DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml",
        name_aircraft="Hybrid Kodiak 100",
    )

    fig.show()


def test_lca_bar_chart_absolute_phase():
    fig = lca_impacts_bar_chart_with_phases_absolute(
        SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml",
        name_aircraft="Hybrid Kodiak 100",
    )
    fig.update_layout(title_text=None)

    fig.show()
    fig.update_layout(height=800, width=1600)
    fig.write_image(RESULT_FOLDER_PATH / "ref_kodiak_component_contribution.pdf")
