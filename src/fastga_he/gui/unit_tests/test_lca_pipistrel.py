#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib

from ..lca_impact import (
    lca_impacts_bar_chart_simple,
    lca_impacts_bar_chart_normalised_weighted,
    lca_impacts_bar_chart_with_contributors,
    lca_impacts_bar_chart_with_components_absolute,
    lca_impacts_search_table,
    lca_raw_impact_comparison,
    lca_raw_impact_comparison_advanced,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca_pipistrel"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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
