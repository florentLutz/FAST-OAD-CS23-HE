# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pytest

from ..a_airframe.a1_wing_weight_analytical import ComputeWingMassAnalytical
from ..b_propulsion.b1_power_train_mass import PowerTrainMass
from ..payload import ComputePayloadForRetrofit

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "data.xml"


def test_propulsion_weight():
    """Tests propulsion weight computation from sample XML data."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(PowerTrainMass()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PowerTrainMass(), ivc)
    weight_b = problem.get_val("data:weight:propulsion:mass", units="kg")
    assert weight_b == pytest.approx(320.00, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_wing_mass_analytical():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingMassAnalytical()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMassAnalytical(), ivc)
    assert problem["data:weight:airframe:wing:mass"] == pytest.approx(172.0, abs=1e-2)


def test_payload_mass():
    """Tests propulsion weight computation from sample XML data."""

    inputs_list = [
        "data:weight:aircraft:target_MTOW",
        "data:weight:aircraft:OWE",
        "data:mission:sizing:fuel",
        "data:weight:aircraft:max_payload",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePayloadForRetrofit(), ivc)
    weight_b = problem.get_val("data:weight:aircraft:payload", units="kg")
    assert weight_b == pytest.approx(619.00, abs=1e-2)

    problem.check_partials(compact_print=True)
