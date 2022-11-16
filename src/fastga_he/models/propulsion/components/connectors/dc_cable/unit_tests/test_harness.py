# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest
import openmdao.api as om

from stdatm import Atmosphere

from ..components.sizing_material_core import SizingMaterialCore
from ..components.sizing_current_per_cable import SizingCurrentPerCable
from ..components.sizing_cable_gauge import SizingCableGauge
from ..components.sizing_resistance_per_length import SizingResistancePerLength
from ..components.sizing_insulation_thickness import SizingInsulationThickness
from ..components.sizing_sheath_thickness import SizingCableSheathThickness
from ..components.sizing_mass_per_length import SizingMassPerLength
from ..components.sizing_contactor_mass import SizingHarnessContactorMass
from ..components.sizing_harness_mass import SizingHarnessMass
from ..components.sizing_reference_resistance import SizingReferenceResistance
from ..components.sizing_heat_capacity_per_length import SizingHeatCapacityPerLength
from ..components.sizing_heat_capacity import SizingHeatCapacityCable
from ..components.sizing_cable_radius import SizingCableRadius
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
    ivc = get_indep_var_comp(
        list_inputs(SizingMaterialCore(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMaterialCore(harness_id="harness_1"), ivc)
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
        list_inputs(SizingCurrentPerCable(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingCurrentPerCable(harness_id="harness_1"), ivc)
    assert problem[
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:current"
    ] == pytest.approx(800.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_diameter():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCableGauge(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingCableGauge(harness_id="harness_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:conductor:section", units="mm*mm"
    ) == pytest.approx(43.44, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:conductor:radius", units="m"
    ) == pytest.approx(3.71e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingResistancePerLength(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingResistancePerLength(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:resistance_per_length",
            units="ohm/km",
        )
        == pytest.approx(0.397, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_insulation_thickness():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInsulationThickness(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingInsulationThickness(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:insulation:thickness",
            units="m",
        )
        == pytest.approx(0.0012, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_sheath_thickness():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCableSheathThickness(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingCableSheathThickness(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:sheath:thickness",
            units="mm",
        )
        == pytest.approx(1.36, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_mass_per_length():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingMassPerLength(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMassPerLength(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:mass_per_length",
            units="kg/m",
        )
        == pytest.approx(0.537, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_contactor_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHarnessContactorMass(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHarnessContactorMass(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:contactor:mass",
            units="kg",
        )
        == pytest.approx(4.95, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingReferenceResistance(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingReferenceResistance(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:resistance",
            units="ohm",
        )
        == pytest.approx(0.00333, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_heat_capacity_per_length():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatCapacityPerLength(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHeatCapacityPerLength(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable"
            ":heat_capacity_per_length",
            units="J/degK/m",
        )
        == pytest.approx(376.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_cable_radius():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCableRadius(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingCableRadius(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:radius",
            units="mm",
        )
        == pytest.approx(6.47, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_heat_capacity():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatCapacityCable(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHeatCapacityCable(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:heat_capacity",
            units="J/degK",
        )
        == pytest.approx(3158.4, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_cable_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHarnessMass(harness_id="harness_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHarnessMass(harness_id="harness_1"), ivc)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass",
            units="kg",
        )
        == pytest.approx(9.46, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_perf_current():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesCurrent(harness_id="harness_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("voltage_in", units="V", val=np.linspace(780, 800, NB_POINTS_TEST))
    ivc.add_output("voltage_out", units="V", val=np.linspace(760, 770, NB_POINTS_TEST))
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
        [0.10076, 0.09323, 0.08571, 0.07822, 0.07076, 0.06331, 0.05588, 0.04847, 0.04108, 0.03372]
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
        [0.10076, 0.09323, 0.08571, 0.07822, 0.07076, 0.06331, 0.05588, 0.04847, 0.04108, 0.03372]
    )
    ivc.add_output("cable_temperature_time_derivative", units="degK/s", val=derivative)
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 300.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperatureIncrease(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_increase = np.array(
        [30.228, 27.969, 25.713, 23.466, 21.228, 18.993, 16.764, 14.541, 12.324, 10.116]
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
    increase = np.array(
        [30.228, 27.969, 25.713, 23.466, 21.228, 18.993, 16.764, 14.541, 12.324, 10.116]
    )
    ivc.add_output("cable_temperature_increase", units="degK", val=increase)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperature(harness_id="harness_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_temperature = np.array(
        [288.15, 316.119, 341.832, 365.298, 386.526, 405.519, 422.283, 436.824, 449.148, 459.264]
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
        [288.15, 316.119, 341.832, 365.298, 386.526, 405.519, 422.283, 436.824, 449.148, 459.264]
    )
    ivc.add_output("cable_temperature", units="degK", val=temperature)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesResistance(harness_id="harness_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_resistance = np.array(
        [0.00326, 0.00363, 0.00397, 0.00427, 0.00455, 0.0048, 0.00502, 0.00521, 0.00537, 0.0055]
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
    ivc.add_output("voltage_in", units="V", val=np.linspace(800, 800, NB_POINTS_TEST))
    ivc.add_output("voltage_out", units="V", val=np.linspace(799, 799, NB_POINTS_TEST))
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
        [288.2, 316.8, 333.7, 343.7, 349.4, 352.5, 354.0, 354.4, 354.3, 353.7]
    )
    expected_resistance = np.array(
        [0.00326, 0.00364, 0.00386, 0.00399, 0.00407, 0.00411, 0.00413, 0.00413, 0.00413, 0.00412]
    )

    assert problem.get_val("temperature.cable_temperature", units="degK") == pytest.approx(
        expected_temperature, rel=1e-2
    )
    assert problem.get_val("resistance.resistance_per_cable", units="ohm") == pytest.approx(
        expected_resistance, rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:current_max", units="A"
    ) == pytest.approx(306.0, rel=1e-2)
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
    ) == pytest.approx(9.48, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_cable():

    ivc = om.IndepVarComp()

    ivc.add_output(name="data:propulsion:he_power_train:DC_cable_harness:harness_1:material", val=1)
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:current_max",
        val=800.0,
        units="A",
    )
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:voltage_max",
        val=1000.0,
        units="V",
    )
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:number_cables", val=1
    )
    ivc.add_output(
        name="data:propulsion:he_power_train:DC_cable_harness:harness_1:length", val=7.0, units="m"
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingHarness(harness_id="harness_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:mass_per_length",
            units="kg/m",
        )
        == pytest.approx(0.540, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:resistance",
            units="ohm",
        )
        == pytest.approx(0.00334, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:heat_capacity",
            units="J/degK",
        )
        == pytest.approx(3169.777, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_cable_harness:harness_1:cable:radius",
            units="mm",
        )
        == pytest.approx(6.478, rel=1e-2)
    )
