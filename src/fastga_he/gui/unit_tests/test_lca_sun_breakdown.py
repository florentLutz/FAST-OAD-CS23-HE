# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

from ..lca_impact import lca_impacts_sun_breakdown

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_lca_sun_breakdown_tbm900():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(pth.join(DATA_FOLDER_PATH, "tbm900_lca_ef.xml"))

    fig.show()

    fig = lca_impacts_sun_breakdown(
        pth.join(DATA_FOLDER_PATH, "tbm900_lca_ef.xml"), full_burst=True
    )

    fig.show()


def test_lca_sun_breakdown_pipistrel():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(pth.join(DATA_FOLDER_PATH, "pipistrel_lca_ef.xml"))

    fig.show()

    fig = lca_impacts_sun_breakdown(
        pth.join(DATA_FOLDER_PATH, "pipistrel_lca_ef.xml"), full_burst=True
    )

    fig.show()


def test_lca_sun_breakdown_kodiak_and_hybrid():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(
        [
            pth.join(DATA_FOLDER_PATH, "kodiak_100_ef.xml"),
            pth.join(DATA_FOLDER_PATH, "hybrid_kodiak_100_ef.xml"),
        ],
        full_burst=True,
        name_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )

    fig.show()


def test_lca_sun_breakdown_kodiak_rel_absolute():
    fig = lca_impacts_sun_breakdown(
        pth.join(DATA_FOLDER_PATH, "kodiak_100_ef.xml"),
        full_burst=True,
        name_aircraft="Reference Kodiak 100",
        rel="single_score",
    )

    fig.show()


def test_lca_sun_breakdown_kodiak_rel_parent():
    fig = lca_impacts_sun_breakdown(
        pth.join(DATA_FOLDER_PATH, "kodiak_100_ef.xml"),
        full_burst=True,
        name_aircraft="Reference Kodiak 100",
        rel="parent",
    )

    fig.show()


def test_lca_sun_breakdown_kodiak_rel_parent_using_list():
    fig = lca_impacts_sun_breakdown(
        [pth.join(DATA_FOLDER_PATH, "kodiak_100_ef.xml")],
        full_burst=True,
        name_aircraft=["Reference Kodiak 100"],
        rel="parent",
    )

    fig.show()


def test_lca_sun_breakdown_kodiak_recipe():
    fig = lca_impacts_sun_breakdown(
        [pth.join(DATA_FOLDER_PATH, "kodiak_100_recipe_end.xml")],
        full_burst=True,
        name_aircraft=["Reference Kodiak 100 using ReCiPe"],
        rel="single_score",
    )

    fig.show()


def test_compare_kodiak_recipe_and_ef():
    # For ReCiPe, it uses equivalent weighting factor to transfer single score to midpoint
    # indicators. Results to be taken with care.
    fig = lca_impacts_sun_breakdown(
        [
            pth.join(DATA_FOLDER_PATH, "kodiak_100_recipe_mid.xml"),
            pth.join(DATA_FOLDER_PATH, "kodiak_100_ef.xml"),
        ],
        full_burst=True,
        name_aircraft=["Reference Kodiak 100 using ReCiPe", "Reference Kodiak 100 using EF"],
        # rel="single_score",
    )

    fig.show()


def test_kodiak_recipe_endpoint():
    fig = lca_impacts_sun_breakdown(
        [
            pth.join(DATA_FOLDER_PATH, "kodiak_100_recipe_end.xml"),
        ],
        full_burst=True,
        name_aircraft=["Reference Kodiak 100 using ReCiPe"],
        rel="single_score",
    )

    fig.show()

    # Uses equivalent weighting factor to transfer single score to midpoint indicators.
    fig2 = lca_impacts_sun_breakdown(
        [
            pth.join(DATA_FOLDER_PATH, "kodiak_100_recipe_mid.xml"),
        ],
        full_burst=True,
        name_aircraft=["Reference Kodiak 100 using ReCiPe"],
        rel="single_score",
    )

    fig2.show()


def test_kodiak_and_hybrid_recipe_endpoint():
    fig = lca_impacts_sun_breakdown(
        [
            pth.join(DATA_FOLDER_PATH, "kodiak_100_recipe_end.xml"),
            pth.join(DATA_FOLDER_PATH, "hybrid_kodiak_recipe_end.xml"),
        ],
        full_burst=True,
        name_aircraft=["Reference Kodiak 100 using ReCiPe", "Hybrid Kodiak 100 using ReCiPe"],
        rel="single_score",
    )

    fig.show()
