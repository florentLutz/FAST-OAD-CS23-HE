# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.perf_maximum import PerformancesMaximum
from ..components.sizing_bus_cross_section_area import SizingBusBarCrossSectionArea
from ..components.sizing_bus_bar_cross_section_dimensions import SizingBusBarCrossSectionDimensions
from ..components.sizing_insulation_thickness import SizingBusBarInsulationThickness
from ..components.sizing_bus_dimensions import SizingBusBarDimensions
from ..components.sizing_bus_bar_weight import SizingBusBarWeight
from ..components.sizing_conductor_self_inductance import SizingBusBarSelfInductance
from ..components.sizing_conductor_mutual_inductance import SizingBusBarMutualInductance
from ..components.sizing_dc_bus_cg_x import SizingDCBusCGX
from ..components.sizing_dc_bus_cg_y import SizingDCBusCGY
from ..components.sizing_dc_bus import SizingDCBus

from ..components.pre_lca_prod_weight_per_fu import PreLCADCBusProdWeightPerFU

from ..components.cstr_enforce import ConstraintsCurrentEnforce, ConstraintsVoltageEnforce
from ..components.cstr_ensure import ConstraintsCurrentEnsure, ConstraintsVoltageEnsure

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_dc_bus.xml"
NB_POINTS_TEST = 10

# Note: The PerformancesElectricalNode component cannot be easily tested on its own as it is
# meant to "direct" the current in the global simulation, so it won't be tested here but rather
# in the assemblies


def test_maximum():
    ivc = om.IndepVarComp()
    voltage = np.array([539.1, 506.3, 588.2, 425.0, 572.7, 512.8, 483.4, 466.9, 497.9, 511.4])
    ivc.add_output("dc_voltage", units="V", val=voltage)
    current_1 = np.array([385.5, 430.0, 334.8, 494.6, 470.2, 427.3, 490.3, 468.6, 368.9, 354.2])
    ivc.add_output("dc_current_in_1", units="A", val=current_1)
    current_2 = np.array([356.5, 360.1, 345.2, 346.6, 470.2, 482.8, 332.1, 388.2, 434.4, 428.0])
    ivc.add_output("dc_current_in_2", units="A", val=current_2)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(
            dc_bus_id="dc_bus_1", number_of_points=NB_POINTS_TEST, number_of_inputs=2
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:voltage_max", units="V"
    ) == pytest.approx(588.2, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:current_max", units="A"
    ) == pytest.approx(940.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_cross_section_area():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBusBarCrossSectionArea(dc_bus_id="dc_bus_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingBusBarCrossSectionArea(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:cross_section:area", units="cm**2"
    ) == pytest.approx(1.90, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_cross_section_dimensions():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBusBarCrossSectionDimensions(dc_bus_id="dc_bus_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingBusBarCrossSectionDimensions(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:cross_section:thickness", units="cm"
    ) == pytest.approx(0.31, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:cross_section:width", units="cm"
    ) == pytest.approx(6.16, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_insulation_thickness():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBusBarInsulationThickness(dc_bus_id="dc_bus_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingBusBarInsulationThickness(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:insulation:thickness", units="cm"
    ) == pytest.approx(0.111, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_dimensions():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBusBarDimensions(dc_bus_id="dc_bus_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingBusBarDimensions(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:length", units="cm"
    ) == pytest.approx(30.222, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:height", units="cm"
    ) == pytest.approx(0.953, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:width", units="cm"
    ) == pytest.approx(6.382, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBusBarWeight(dc_bus_id="dc_bus_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingBusBarWeight(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(1.12, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_self_inductance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBusBarSelfInductance(dc_bus_id="dc_bus_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingBusBarSelfInductance(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:conductor:self_inductance", units="nH"
    ) == pytest.approx(166.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bus_bar_mutual_inductance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingBusBarMutualInductance(dc_bus_id="dc_bus_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingBusBarMutualInductance(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:conductor:mutual_inductance", units="nH"
    ) == pytest.approx(1271.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_dc_bus_cg_x():
    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingDCBusCGX(dc_bus_id="dc_bus_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingDCBusCGX(dc_bus_id="dc_bus_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:DC_bus:dc_bus_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_dc_bus_cg_y():
    expected_cg = [2.04, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingDCBusCGY(dc_bus_id="dc_bus_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingDCBusCGY(dc_bus_id="dc_bus_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:DC_bus:dc_bus_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_dc_bus_bar_sizing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(SizingDCBus(dc_bus_id="dc_bus_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCBus(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:conductor:self_inductance", units="nH"
    ) == pytest.approx(171.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:conductor:mutual_inductance", units="nH"
    ) == pytest.approx(1273.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:CG:y", units="m"
    ) == pytest.approx(2.04, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:low_speed:CD0",
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:cruise:CD0",
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_current_enforce():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnforce(dc_bus_id="dc_bus_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsCurrentEnforce(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:current_caliber", units="A"
    ) == pytest.approx(800.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_enforce():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(dc_bus_id="dc_bus_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnforce(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:voltage_caliber", units="V"
    ) == pytest.approx(500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_current_ensure():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnsure(dc_bus_id="dc_bus_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsCurrentEnsure(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:DC_bus:dc_bus_1:current_caliber", units="A"
    ) == pytest.approx(-141.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_ensure():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(dc_bus_id="dc_bus_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnsure(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:DC_bus:dc_bus_1:voltage_caliber", units="V"
    ) == pytest.approx(-88.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass",
        "data:environmental_impact:aircraft_per_fu",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCADCBusProdWeightPerFU(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass_per_fu", units="kg"
    ) == pytest.approx(1.12e-6, rel=1e-3)

    problem.check_partials(compact_print=True)
