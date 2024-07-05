# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import pytest

from ..payload_range import payload_range_outer, payload_range_inner

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_payload_range():
    """
    Basic tests for payload range display.
    """

    # Check that we can create a plot with no previous plot
    fig = payload_range_outer(
        pth.join(DATA_FOLDER_PATH, "sample_payload_range_fuel.xml"), name="Fuel"
    )

    # Check that we can superimpose two payload range diagram
    fig = payload_range_outer(
        pth.join(DATA_FOLDER_PATH, "sample_payload_range_hybrid.xml"), name="Hybrid", fig=fig
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_payload_range_electric():

    """
    Tests for payload range display with electric.
    """

    fig = payload_range_outer(
        pth.join(DATA_FOLDER_PATH, "sample_payload_range_electric.xml"), name="Electric"
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_inner_payload_range_hybrid():
    """
    Tests for inner payload range display with hybrid.
    """

    fig = payload_range_inner(
        pth.join(DATA_FOLDER_PATH, "sample_payload_range_hybrid.xml"),
        smooth=True,
        grid_accuracy=1000,
        z_filter=10,
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_inner_payload_range_comparison():
    """
    Tests for inner payload range display with hybrid.
    """

    fig = payload_range_inner(
        ref_aircraft_file_path=pth.join(DATA_FOLDER_PATH, "sample_payload_range_fuel.xml"),
        sec_aircraft_file_path=pth.join(DATA_FOLDER_PATH, "sample_payload_range_hybrid.xml"),
        smooth=True,
        grid_accuracy=1000,
        z_filter=5,
    )

    fig.show()
