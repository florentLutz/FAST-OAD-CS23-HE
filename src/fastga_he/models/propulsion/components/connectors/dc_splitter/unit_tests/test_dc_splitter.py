# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_dc_splitter_cg import SizingDCSplitterCG
from ..components.perf_mission_power_split import PerformancesMissionPowerSplit

from ..components.sizing_dc_splitter import SizingDCSplitter

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_dc_splitter.xml"
NB_POINTS_TEST = 10


def test_dc_sspc_cg():

    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingDCSplitterCG(dc_splitter_id="dc_splitter_id_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingDCSplitterCG(dc_splitter_id="dc_splitter_id_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:DC_splitter:dc_splitter_id_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_perf_power_split_formatting():

    power_split_float = 42.0
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_split",
        val=power_split_float,
        units="percent",
    )

    problem = run_system(
        PerformancesMissionPowerSplit(
            dc_splitter_id="dc_splitter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    power_split_output = problem.get_val(
        "power_split",
        units="percent",
    )
    expected_power_split = np.full(NB_POINTS_TEST, power_split_float)
    assert power_split_output == pytest.approx(expected_power_split, rel=1e-4)

    # Let's now try with a full power split
    power_split_array = np.linspace(60, 40, NB_POINTS_TEST)
    ivc_2 = om.IndepVarComp()
    ivc_2.add_output(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_split",
        val=power_split_array,
        units="percent",
    )

    problem = run_system(
        PerformancesMissionPowerSplit(
            dc_splitter_id="dc_splitter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc_2,
    )
    power_split_output = problem.get_val(
        "power_split",
        units="percent",
    )
    assert power_split_output == pytest.approx(power_split_array, rel=1e-4)


def test_sizing_dc_splitter():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSplitter(dc_splitter_id="dc_splitter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingDCSplitter(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:mass", units="kg"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)
