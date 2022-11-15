# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.perf_module_current import PerformancesModuleCurrent
from ..components.perf_open_circuit_voltage import PerformancesOpenCircuitVoltage
from ..components.perf_internal_resistance import PerformancesInternalResistance
from ..components.perf_cell_voltage import PerformancesCellVoltage
from ..components.perf_module_voltage import PerformancesModuleVoltage
from ..components.perf_battery_voltage import PerformancesBatteryVoltage

from ..components.perf_battery_pack import PerformancesBatteryPack

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_battery_pack.xml"
NB_POINTS_TEST = 10


def test_current_module():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesModuleCurrent(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    current_out = np.linspace(400, 410, NB_POINTS_TEST)
    ivc.add_output("current_out", current_out, units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesModuleCurrent(
            number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
        ),
        ivc,
    )
    assert problem.get_val("current_one_module", units="A") == pytest.approx(
        [10.0, 10.03, 10.06, 10.08, 10.11, 10.14, 10.17, 10.19, 10.22, 10.25], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_open_circuit_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output("state_of_charge", val=np.linspace(100, 40, NB_POINTS_TEST), units="percent")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesOpenCircuitVoltage(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("open_circuit_voltage", units="V") == pytest.approx(
        [4.04, 3.96, 3.89, 3.84, 3.79, 3.75, 3.72, 3.68, 3.65, 3.62], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_internal_resistance():

    ivc = om.IndepVarComp()
    ivc.add_output("state_of_charge", val=np.linspace(100, 40, NB_POINTS_TEST), units="percent")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesInternalResistance(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("internal_resistance", units="ohm") * 1e3 == pytest.approx(
        [2.96, 3.44, 3.83, 4.13, 4.35, 4.51, 4.62, 4.68, 4.71, 4.71], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_internal_resistance():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "current_one_module",
        units="A",
        val=np.array([10.0, 10.03, 10.06, 10.08, 10.11, 10.14, 10.17, 10.19, 10.22, 10.25]),
    )
    ivc.add_output(
        "open_circuit_voltage",
        units="V",
        val=np.array([4.04, 3.96, 3.89, 3.84, 3.79, 3.75, 3.72, 3.68, 3.65, 3.62]),
    )
    ivc.add_output(
        "internal_resistance",
        units="ohm",
        val=np.array([2.96, 3.44, 3.83, 4.13, 4.35, 4.51, 4.62, 4.68, 4.71, 4.71]) * 1e-3,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCellVoltage(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("terminal_voltage", units="V") == pytest.approx(
        [4.01, 3.93, 3.85, 3.8, 3.75, 3.7, 3.67, 3.63, 3.6, 3.57], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_module_voltage():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesModuleVoltage(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "terminal_voltage",
        units="V",
        val=np.array([4.01, 3.93, 3.85, 3.8, 3.75, 3.7, 3.67, 3.63, 3.6, 3.57]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesModuleVoltage(
            number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
        ),
        ivc,
    )
    assert problem.get_val("module_voltage", units="V") == pytest.approx(
        [802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_battery_voltage():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "module_voltage",
        val=np.array([802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0]),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBatteryVoltage(number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("voltage_out", units="V") == pytest.approx(
        [802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performances_battery_pack():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesBatteryPack(
                number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    current_out = np.linspace(400, 410, NB_POINTS_TEST)
    ivc.add_output("current_out", current_out, units="A")
    ivc.add_output("state_of_charge", val=np.linspace(100, 40, NB_POINTS_TEST), units="percent")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesBatteryPack(number_of_points=NB_POINTS_TEST, battery_pack_id="battery_pack_1"),
        ivc,
    )
    assert problem.get_val("voltage_out", units="V") == pytest.approx(
        [802.0, 786.0, 770.0, 760.0, 750.0, 740.0, 734.0, 726.0, 720.0, 714.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)
