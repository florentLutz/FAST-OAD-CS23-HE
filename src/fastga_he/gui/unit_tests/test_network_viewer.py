# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os
from shutil import rmtree

import pytest

from ..power_train_network_viewer import power_train_network_viewer

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")
RESULTS_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_pt_network_viewer(cleanup):
    """
    Basic tests for testing the power train weight breakdown.
    """

    # Create a directory to save graph to
    os.makedirs(RESULTS_FOLDER_PATH)

    # No real way to verify the plot, we wil just check that it is created.
    power_train_network_viewer(
        os.path.join(DATA_FOLDER_PATH, "simple_assembly.yml"),
        os.path.join(RESULTS_FOLDER_PATH, "network.html"),
    )

    assert os.path.exists(os.path.join(RESULTS_FOLDER_PATH, "network.html"))

    # Cleanup to avoid any over-clogging
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_pt_network_viewer_tri_prop(cleanup):
    """
    Basic tests for testing the power train weight breakdown.
    """

    # Create a directory to save graph to
    os.makedirs(RESULTS_FOLDER_PATH)

    # No real way to verify the plot, we wil just check that it is created.
    power_train_network_viewer(
        os.path.join(DATA_FOLDER_PATH, "simple_assembly_tri_prop.yml"),
        os.path.join(RESULTS_FOLDER_PATH, "network.html"),
    )

    assert os.path.exists(os.path.join(RESULTS_FOLDER_PATH, "network.html"))

    # Cleanup to avoid any over-clogging
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_pt_network_viewer_tri_prop_two_chainz(cleanup):
    """
    Basic tests for testing the power train weight breakdown.
    """

    # Create a directory to save graph to
    os.makedirs(RESULTS_FOLDER_PATH)

    # No real way to verify the plot, we wil just check that it is created.
    power_train_network_viewer(
        os.path.join(DATA_FOLDER_PATH, "simple_assembly_tri_prop_two_chainz.yml"),
        os.path.join(RESULTS_FOLDER_PATH, "network.html"),
    )

    assert os.path.exists(os.path.join(RESULTS_FOLDER_PATH, "network.html"))

    # Cleanup to avoid any over-clogging
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
