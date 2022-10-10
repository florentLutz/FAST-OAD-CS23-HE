# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest
import openmdao.api as om

from stdatm import Atmosphere

from ..components.sizing_material_core import MaterialCore
from ..components.sizing_current_per_cable import CurrentPerCable
from ..components.sizing_cable_gauge import CableGauge
from ..components.sizing_resistance_per_length import ResistancePerLength
from ..components.sizing_insulation_thickness import InsulationThickness
from ..components.sizing_mass_per_length import MassPerLength
from ..components.sizing_harness_mass import HarnessMass
from ..components.sizing_reference_resistance import ReferenceResistance
from ..components.sizing_heat_capacity_per_length import HeatCapacityPerLength
from ..components.sizing_heat_capacity import HeatCapacityCable
from ..components.sizing_cable_radius import CableRadius
from ..components.perf_current import PerformancesCurrent, PerformancesHarnessCurrent
from ..components.perf_temperature_derivative import PerformancesTemperatureDerivative
from ..components.perf_temperature_increase import PerformancesTemperatureIncrease
from ..components.perf_temperature import PerformancesTemperature
from ..components.perf_resistance import PerformancesResistance

from ..components.perf_harness import PerformanceHarness
from ..components.sizing_harness import SizingHarness

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_cable.xml"
NB_POINTS_TEST = 10


def test_material_core():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(MaterialCore(harness_id="harness_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(MaterialCore(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:properties:density",
            units="kg/m**3",
        )
        == pytest.approx(8960.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:properties:specific_heat",
            units="J/degK/kg",
        )
        == pytest.approx(386.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:properties:resistance_temperature_scale_factor",
            units="degK**-1",
        )
        == pytest.approx(0.00393, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_compute_current_per_cable():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(CurrentPerCable(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CurrentPerCable(harness_id="harness_1"), ivc)
    assert problem[
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:current"
    ] == pytest.approx(47.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_diameter():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:current",
        val=110.0,
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = om.Problem()
    model = problem.model
    model.add_subsystem("ivc", ivc, promotes=["data:*"])
    model.add_subsystem(
        "diameter_computation", CableGauge(harness_id="harness_1"), promotes=["data:*"]
    )
    problem.setup()
    problem.run_model()
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:conductor:section", units="mm*mm"
    ) == pytest.approx(5.02, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:conductor:radius", units="m"
    ) == pytest.approx(1.26e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ResistancePerLength(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ResistancePerLength(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:resistance_per_length",
            units="ohm/km",
        )
        == pytest.approx(3.44, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_insulation_thickness():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(InsulationThickness(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(InsulationThickness(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:insulation:thickness",
            units="m",
        )
        == pytest.approx(0.327e-2, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_mass_per_length():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(MassPerLength(harness_id="harness_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(MassPerLength(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:mass_per_length",
            units="kg/m",
        )
        == pytest.approx(0.429, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ReferenceResistance(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ReferenceResistance(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:resistance",
            units="ohm",
        )
        == pytest.approx(0.0263, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_heat_capacity_per_length():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(HeatCapacityPerLength(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(HeatCapacityPerLength(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable"
            ":heat_capacity_per_length",
            units="J/degK/m",
        )
        == pytest.approx(551.37, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_cable_radius():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(CableRadius(harness_id="harness_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CableRadius(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:radius",
            units="mm",
        )
        == pytest.approx(7.87, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_heat_capacity():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(HeatCapacityCable(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(HeatCapacityCable(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:heat_capacity",
            units="J/degK",
        )
        == pytest.approx(6422.976, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_cable_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(HarnessMass(harness_id="harness_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(HarnessMass(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass",
            units="kg",
        )
        == pytest.approx(9.07, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_perf_current():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesCurrent(harness_id="harness_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("voltage_b", units="V", val=np.linspace(780, 800, NB_POINTS_TEST))
    ivc.add_output("voltage_a", units="V", val=np.linspace(760, 770, NB_POINTS_TEST))
    ivc.add_output("resistance_per_cable", units="ohm", val=np.full(NB_POINTS_TEST, 0.0263))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCurrent(harness_id="harness_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_current = np.array(
        [760.46, 802.7, 844.95, 887.2, 929.45, 971.69, 1013.94, 1056.19, 1098.44, 1140.68]
    )
    assert (
        problem.get_val(
            "current",
            units="A",
        )
        == pytest.approx(expected_current, rel=1e-2)
    )

    # Don't mind the error on the number of cable, due to fd computation, no actual dependency

    problem.check_partials(compact_print=True)


def test_perf_tot_current():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesHarnessCurrent(harness_id="harness_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    current = np.array(
        [760.46, 802.7, 844.95, 887.2, 929.45, 971.69, 1013.94, 1056.19, 1098.44, 1140.68]
    )
    ivc.add_output("current", units="A", val=current)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHarnessCurrent(harness_id="harness_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_current = np.array(
        [760.46, 802.7, 844.95, 887.2, 929.45, 971.69, 1013.94, 1056.19, 1098.44, 1140.68]
    )
    assert (
        problem.get_val(
            "total_current",
            units="A",
        )
        == pytest.approx(expected_current, rel=1e-2)
    )

    # Don't mind the error on the number of cable, due to fd computation, no actual dependency

    problem.check_partials(compact_print=True)


def test_perf_temperature_derivative():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTemperatureDerivative(
                harness_id="harness_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("resistance_per_cable", units="ohm", val=np.full(NB_POINTS_TEST, 0.0263))
    ivc.add_output(
        "current",
        units="A",
        val=np.linspace(110.0, 100.0, NB_POINTS_TEST),
    )
    ivc.add_output(
        "heat_transfer_coefficient",
        units="W/cm**2/degK",
        val=np.full(NB_POINTS_TEST, 0.0011),
    )
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=np.full(NB_POINTS_TEST, 288.15),
    )
    ivc.add_output(
        "cable_temperature",
        units="degK",
        val=np.linspace(288.15, 338.15, NB_POINTS_TEST),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperatureDerivative(harness_id="harness_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )
    expected_derivative = np.array(
        [0.04955, 0.04492, 0.04031, 0.0357, 0.03111, 0.02653, 0.02195, 0.01739, 0.01284, 0.00829]
    )
    assert (
        problem.get_val(
            "cable_temperature_time_derivative",
            units="degK/s",
        )
        == pytest.approx(expected_derivative, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_perf_temperature_increase():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesTemperatureIncrease(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    derivative = np.array(
        [0.04955, 0.04492, 0.04031, 0.0357, 0.03111, 0.02653, 0.02195, 0.01739, 0.01284, 0.00829]
    )
    ivc.add_output("cable_temperature_time_derivative", units="degK/s", val=derivative)
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 300.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperatureIncrease(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_increase = np.array(
        [14.865, 13.476, 12.093, 10.71, 9.333, 7.959, 6.585, 5.217, 3.852, 2.487]
    )
    assert (
        problem.get_val(
            "cable_temperature_increase",
            units="degK",
        )
        == pytest.approx(expected_increase, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_perf_temperature_profile():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTemperature(harness_id="harness_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    increase = np.array([14.865, 13.476, 12.093, 10.71, 9.333, 7.959, 6.585, 5.217, 3.852, 2.487])
    ivc.add_output("cable_temperature_increase", units="degK", val=increase)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperature(harness_id="harness_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_temperature = np.array(
        [288.15, 301.626, 313.719, 324.429, 333.762, 341.721, 348.306, 353.523, 357.375, 359.862]
    )
    assert (
        problem.get_val(
            "cable_temperature",
            units="degK",
        )
        == pytest.approx(expected_temperature, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_resistance_profile():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesResistance(harness_id="harness_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    temperature = np.array(
        [288.15, 301.626, 313.719, 324.429, 333.762, 341.721, 348.306, 353.523, 357.375, 359.862]
    )
    ivc.add_output("cable_temperature", units="degK", val=temperature)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesResistance(harness_id="harness_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_resistance = np.array(
        [
            0.0257,
            0.0271,
            0.0284,
            0.0295,
            0.0304,
            0.0313,
            0.0320,
            0.0325,
            0.0329,
            0.0331,
        ]
    )
    assert (
        problem.get_val(
            "resistance_per_cable",
            units="ohm",
        )
        == pytest.approx(expected_resistance, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_performances_harness():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(PerformanceHarness(harness_id="harness_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("voltage_b", units="V", val=np.linspace(800, 800, NB_POINTS_TEST))
    ivc.add_output("voltage_a", units="V", val=np.linspace(798, 798, NB_POINTS_TEST))
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(np.linspace(0, 2000, NB_POINTS_TEST), altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformanceHarness(harness_id="harness_1", number_of_points=NB_POINTS_TEST), ivc
    )

    expected_temperature = np.array(
        [288.1, 296.6, 302.4, 306.3, 308.8, 310.3, 311.0, 311.2, 311.0, 310.5]
    )
    expected_resistance = np.array(
        [0.0257, 0.0266, 0.0272, 0.0276, 0.0279, 0.0280, 0.0281, 0.0281, 0.0281, 0.0281]
    )

    assert problem.get_val("temperature.cable_temperature", units="degK") == pytest.approx(
        expected_temperature, rel=1e-2
    )
    assert problem.get_val("resistance.resistance_per_cable", units="ohm") == pytest.approx(
        expected_resistance, rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:current_max", units="A"
    ) == pytest.approx(77.56, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:voltage_max", units="V"
    ) == pytest.approx(800.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_harness():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHarness(harness_id="harness_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHarness(harness_id="harness_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass", units="kg"
    ) == pytest.approx(2.90, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_cable():

    ivc = om.IndepVarComp()

    ivc.add_output(name="data:propulsion:he_power_train:DC_cable_harness:harness_1:material", val=1)
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:current_max",
        val=200.0,
        units="A",
    )
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:voltage_max",
        val=5000.0,
        units="V",
    )
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:number_cables", val=1
    )
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:length", val=1.0, units="m"
    )
    ivc.add_output(
        name="settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness",
        val=3.3,
        units="mm",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHarness(harness_id="harness_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:mass_per_length",
            units="kg/m",
        )
        == pytest.approx(0.29, rel=1e-2)
    )
