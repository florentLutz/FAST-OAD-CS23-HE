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
from ..components.sizing_dc_bus import SizingDCBus

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

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


def test_dc_bus_bar_sizing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(SizingDCBus(dc_bus_id="dc_bus_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingDCBus(dc_bus_id="dc_bus_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(1.12, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:conductor:self_inductance", units="nH"
    ) == pytest.approx(166.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:conductor:mutual_inductance", units="nH"
    ) == pytest.approx(1271.0, rel=1e-2)

    problem.check_partials(compact_print=True)
