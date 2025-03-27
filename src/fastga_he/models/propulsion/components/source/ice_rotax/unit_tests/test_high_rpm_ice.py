# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from stdatm import Atmosphere

from ..components.cstr_enforce import ConstraintsSeaLevelPowerEnforce
from ..components.cstr_ensure import ConstraintsSeaLevelPowerEnsure
from ..components.cstr_high_rpm_ice import ConstraintHighRPMICEPowerRateMission

from ..components.sizing_displacement_volume import SizingHighRPMICEDisplacementVolume

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_high_rpm_ice.xml"
NB_POINTS_TEST = 10


def test_constraint_power_enforce():
    ivc = get_indep_var_comp(
        ["data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_max_SL"], __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsSeaLevelPowerEnforce(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL", units="kW"
    ) == pytest.approx(68.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_ensure():
    ivc = get_indep_var_comp(
        [
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_max_SL",
            "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL",
        ],
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsSeaLevelPowerEnsure(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL", units="kW"
    ) == pytest.approx(-1.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = get_indep_var_comp(
        ["data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL"], __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintHighRPMICEPowerRateMission(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:shaft_power_rating", units="kW"
    ) == pytest.approx(69.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_displacement_volume():
    inputs_list = [
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:power_rating_SL",
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:pme_max",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHighRPMICEDisplacementVolume(high_rpm_ice_id="ice_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:high_rpm_ICE:ice_1:displacement_volume", units="cm**3"
    ) == pytest.approx(1352.0, rel=1e-2)

    problem.check_partials(compact_print=True)
