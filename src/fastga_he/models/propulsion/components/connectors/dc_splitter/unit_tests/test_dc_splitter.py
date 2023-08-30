# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_dc_splitter_cross_section_area import SizingDCSplitterCrossSectionArea
from ..components.sizing_dc_splitter_cross_section_dimensions import (
    SizingSplitterCrossSectionDimensions,
)
from ..components.sizing_dc_splitter_insulation_thickness import SizingDCSplitterInsulationThickness
from ..components.sizing_dc_splitter_dimensions import SizingDCSplitterDimensions
from ..components.sizing_dc_splitter_weight import SizingDCSplitterWeight
from ..components.sizing_dc_splitter_cg_x import SizingDCSplitterCGX
from ..components.sizing_dc_splitter_cg_y import SizingDCSplitterCGY
from ..components.perf_mission_power_split import PerformancesMissionPowerSplit
from ..components.perf_mission_power_share import PerformancesMissionPowerShare
from ..components.perf_maximum import PerformancesMaximum

from ..components.cstr_enforce import ConstraintsCurrentEnforce, ConstraintsVoltageEnforce
from ..components.cstr_ensure import ConstraintsCurrentEnsure, ConstraintsVoltageEnsure

from ..components.sizing_dc_splitter import SizingDCSplitter

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_dc_splitter.xml"
NB_POINTS_TEST = 10


def test_splitter_cross_section_area():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSplitterCrossSectionArea(dc_splitter_id="dc_splitter_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCSplitterCrossSectionArea(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:cross_section:area", units="cm**2"
    ) == pytest.approx(1.01, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_cross_section_dimensions():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingSplitterCrossSectionDimensions(dc_splitter_id="dc_splitter_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingSplitterCrossSectionDimensions(dc_splitter_id="dc_splitter_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:cross_section:thickness",
            units="cm",
        )
        == pytest.approx(0.224, rel=1e-2)
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:cross_section:width", units="cm"
    ) == pytest.approx(4.49, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_insulation_thickness():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSplitterInsulationThickness(dc_splitter_id="dc_splitter_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCSplitterInsulationThickness(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:insulation:thickness", units="cm"
    ) == pytest.approx(0.113, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_dimensions():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSplitterDimensions(dc_splitter_id="dc_splitter_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCSplitterDimensions(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:length", units="cm"
    ) == pytest.approx(50.226, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:height", units="cm"
    ) == pytest.approx(0.787, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:width", units="cm"
    ) == pytest.approx(4.716, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSplitterWeight(dc_splitter_id="dc_splitter_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCSplitterWeight(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:mass", units="kg"
    ) == pytest.approx(1.027, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_dc_sspc_cg_x():

    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingDCSplitterCGX(dc_splitter_id="dc_splitter_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingDCSplitterCGX(dc_splitter_id="dc_splitter_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_dc_sspc_cg_x():

    expected_cg = [1.87, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingDCSplitterCGY(dc_splitter_id="dc_splitter_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingDCSplitterCGY(dc_splitter_id="dc_splitter_1", position=option), ivc
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_constraints_current_enforce():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnforce(dc_splitter_id="dc_splitter_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsCurrentEnforce(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:current_caliber", units="A"
    ) == pytest.approx(500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_enforce():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(dc_splitter_id="dc_splitter_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnforce(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:voltage_caliber", units="V"
    ) == pytest.approx(635.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_current_ensure():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnsure(dc_splitter_id="dc_splitter_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsCurrentEnsure(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:DC_splitter:dc_splitter_1:current_caliber", units="A"
    ) == pytest.approx(1.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_ensure():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(dc_splitter_id="dc_splitter_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnsure(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:DC_splitter:dc_splitter_1:voltage_caliber", units="V"
    ) == pytest.approx(-37.01, rel=1e-2)

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


def test_perf_power_share_formatting():

    power_split_float = 150.0e3
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_share",
        val=power_split_float,
        units="W",
    )

    problem = run_system(
        PerformancesMissionPowerShare(
            dc_splitter_id="dc_splitter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    power_split_output = problem.get_val(
        "power_share",
        units="W",
    )
    expected_power_split = np.full(NB_POINTS_TEST, power_split_float)
    assert power_split_output == pytest.approx(expected_power_split, rel=1e-4)

    problem.check_partials(compact_print=True)

    # Let's now try with a full power split
    power_split_array = np.linspace(60e3, 40e3, NB_POINTS_TEST)
    ivc_2 = om.IndepVarComp()
    ivc_2.add_output(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_share",
        val=power_split_array,
        units="W",
    )

    problem = run_system(
        PerformancesMissionPowerShare(
            dc_splitter_id="dc_splitter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc_2,
    )
    power_split_output = problem.get_val(
        "power_share",
        units="W",
    )
    assert power_split_output == pytest.approx(power_split_array, rel=1e-4)

    problem.check_partials(compact_print=True)


def test_perf_maximum():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "dc_voltage",
        val=np.linspace(500, 612, NB_POINTS_TEST),
        units="V",
    )
    ivc.add_output(
        "dc_current_out",
        val=np.linspace(120, 345, NB_POINTS_TEST),
        units="A",
    )
    ivc.add_output(
        "dc_current_in_1",
        val=np.linspace(150, 400, NB_POINTS_TEST),
        units="A",
    )
    ivc.add_output(
        "dc_current_in_2",
        val=np.linspace(-30, -55, NB_POINTS_TEST),
        units="A",
    )

    problem = run_system(
        PerformancesMaximum(dc_splitter_id="dc_splitter_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:voltage_max"
    ) == pytest.approx(612, rel=1e-4)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:current_max"
    ) == pytest.approx(400, rel=1e-4)

    problem.check_partials(compact_print=True)


def test_sizing_dc_splitter():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCSplitter(dc_splitter_id="dc_splitter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingDCSplitter(dc_splitter_id="dc_splitter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:mass", units="kg"
    ) == pytest.approx(1.027, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:CG:y", units="m"
    ) == pytest.approx(1.87, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)
