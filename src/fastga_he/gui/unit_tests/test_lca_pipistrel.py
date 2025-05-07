#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib

import pytest

from ..lca_impact import (
    lca_impacts_bar_chart_simple,
    lca_impacts_sun_breakdown,
    lca_impacts_bar_chart_normalised,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca_pipistrel"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_eu_mix.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro (FR mix)",
            "Pipistrel Velis Electro (EU mix)",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sun_breakdown_pipistrel_presentation():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
        ],
        full_burst=True,
        rel="single_score",
        name_aircraft=["Thermal propulsion", "Electric Propulsion"],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out_recipe.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_ef():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro",
        ],
    )

    fig.show()
