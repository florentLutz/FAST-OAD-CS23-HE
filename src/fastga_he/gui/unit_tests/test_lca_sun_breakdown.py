# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import pytest

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
        pth.join(DATA_FOLDER_PATH, "kodiak_100_ef.xml"),
        full_burst=True,
        name_aircraft="Reference Kodiak 100",
    )

    fig.show()

    fig = lca_impacts_sun_breakdown(
        pth.join(DATA_FOLDER_PATH, "hybrid_kodiak_100_ef.xml"),
        full_burst=True,
        name_aircraft="Hybrid Kodiak 100",
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
