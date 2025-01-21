# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import pathlib

import time

import pytest

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..lca import LCA

DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "results" / "parametric_study"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_environmental_impact_function_span():
    t1 = time.time()

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "hybrid_propulsion.yml",
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="flight",
        impact_assessment_method="EF v3.1",
        normalization=True,
        weighting=True,
        aircraft_lifespan_in_hours=True,
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        DATA_FOLDER_PATH / "hybrid_kodiak.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    problem.output_file_path = RESULTS_FOLDER_PATH / "default.xml"
    problem.write_outputs()

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        1.113997499958996e-05, rel=1e-3
    )

    t2 = time.time()

    print("First run", t2 - t1)

    problem.set_val("data:TLAR:max_airframe_hours", val=3524.9, units="h")

    problem.run_model()

    problem.output_file_path = RESULTS_FOLDER_PATH / "reduced_use.xml"
    problem.write_outputs()

    t3 = time.time()

    print("Second run", t3-t2)
