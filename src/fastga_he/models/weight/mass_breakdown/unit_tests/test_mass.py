# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pytest

from ..b_propulsion.b1_power_train_mass import PowerTrainMass

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
