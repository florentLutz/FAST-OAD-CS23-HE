# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_energy_coefficient_scaling import (
    SizingDCDCConverterEnergyCoefficientScaling,
)
from ..components.sizing_energy_coefficients import SizingDCDCConverterEnergyCoefficients
from ..components.sizing_resistance_scaling import SizingDCDCConverterResistanceScaling
from ..components.sizing_reference_resistance import SizingDCDCConverterResistances
from ..components.sizing_inductor_inductance import SizingDCDCConverterInductorInductance
from ..components.sizing_capacitor_capacity import SizingDCDCConverterCapacitorCapacity
from ..components.sizing_weight import SizingDCDCConverterWeight
from ..components.sizing_dc_dc_converter import SizingDCDCConverter
from ..components.perf_load_side import PerformancesConverterLoadSide
from ..components.perf_duty_cycle import PerformancesDutyCycle
from ..components.perf_conduction_losses import PerformancesConductionLosses
from ..components.perf_switching_losses import PerformancesSwitchingLosses
from ..components.perf_total_losses import PerformancesLosses
from ..components.perf_efficiency import PerformancesEfficiency

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_dc_dc_converter.xml"
NB_POINTS_TEST = 10

# Note: The PerformancesConverterRelations, PerformancesConverterGeneratorSide and
# PerformancesDCDCConverter components cannot be easily tested on its own as it is meant to act
# jointly with the other components of the converter, so it won't be tested here but rather in
# the assemblies


def test_energy_coefficients_scaling_ratio():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingDCDCConverterEnergyCoefficientScaling(dc_dc_converter_id="dc_dc_converter_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingDCDCConverterEnergyCoefficientScaling(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:scaling:a"
    ) == pytest.approx(1.125, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:scaling:c"
    ) == pytest.approx(0.888, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_energy_coefficient():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterEnergyCoefficients(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterEnergyCoefficients(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_on:a"
    ) == pytest.approx(0.02379429, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_rr:a"
    ) == pytest.approx(0.00637224, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_off:a"
    ) == pytest.approx(0.02261044, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_on:b"
    ) == pytest.approx(3.326e-05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_rr:b"
    ) == pytest.approx(0.000340, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_off:b"
    ) == pytest.approx(0.000254, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_on:c"
    ) == pytest.approx(3.420e-7, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_rr:c"
    ) == pytest.approx(-3.005e-8, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:energy_off:c"
    ) == pytest.approx(-1.158e-7, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterResistanceScaling(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterResistanceScaling(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:scaling:resistance"
    ) == pytest.approx(1.125, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterResistances(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterResistances(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:igbt:resistance",
            units="ohm",
        )
        == pytest.approx(0.002265, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:resistance",
            units="ohm",
        )
        == pytest.approx(0.002805, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inductor_inductance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterInductorInductance(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterInductorInductance(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:inductor:inductance",
            units="mH",
        )
        == pytest.approx(0.447, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_capacitor_capacity():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterCapacitorCapacity(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingDCDCConverterCapacitorCapacity(dc_dc_converter_id="dc_dc_converter_1"), ivc
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor:capacity",
            units="mF",
        )
        == pytest.approx(0.484, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_converter_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverterWeight(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingDCDCConverterWeight(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass",
            units="kg",
        )
        == pytest.approx(86.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_dc_dc_converter_sizing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingDCDCConverter(dc_dc_converter_id="dc_dc_converter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingDCDCConverter(dc_dc_converter_id="dc_dc_converter_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass",
            units="kg",
        )
        == pytest.approx(86.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:capacitor:capacity",
            units="mF",
        )
        == pytest.approx(0.484, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:diode:resistance",
            units="ohm",
        )
        == pytest.approx(0.002103, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_load_side():

    ivc = om.IndepVarComp()
    power = np.linspace(350, 400, NB_POINTS_TEST)
    ivc.add_output("power", power, units="kW")
    voltage_in = np.full(NB_POINTS_TEST, 810)
    ivc.add_output("dc_voltage_in", voltage_in, units="V")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesConverterLoadSide(number_of_points=NB_POINTS_TEST), ivc)

    expected_current = np.array(
        [432.1, 439.0, 445.8, 452.7, 459.5, 466.4, 473.3, 480.1, 487.0, 493.8]
    )
    assert problem.get_val("dc_current_in", units="A") == pytest.approx(expected_current, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_duty_cycle():

    ivc = om.IndepVarComp()
    voltage_out = np.linspace(710, 910, NB_POINTS_TEST)
    ivc.add_output("dc_voltage_out", voltage_out, units="V")
    voltage_in = np.full(NB_POINTS_TEST, 810)
    ivc.add_output("dc_voltage_in", voltage_in, units="V")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesDutyCycle(number_of_points=NB_POINTS_TEST), ivc)

    expected_duty_cycle = np.array(
        [0.467, 0.475, 0.482, 0.489, 0.497, 0.503, 0.51, 0.517, 0.523, 0.529]
    )
    assert problem.get_val("duty_cycle") == pytest.approx(expected_duty_cycle, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_conduction_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesConductionLosses(
                dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    duty_cycle = np.array([0.467, 0.475, 0.482, 0.489, 0.497, 0.503, 0.51, 0.517, 0.523, 0.529])
    ivc.add_output("duty_cycle", duty_cycle)
    dc_current_out = np.linspace(200.0, 400.0, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesConductionLosses(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    expected_losses_igbt = np.array(
        [148.9, 192.8, 243.1, 301.6, 371.3, 446.4, 534.6, 634.6, 743.0, 864.2]
    )
    assert problem.get_val("conduction_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [470.5, 552.7, 641.3, 737.0, 841.0, 950.7, 1069.4, 1196.4, 1330.4, 1472.9]
    )
    assert problem.get_val("conduction_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )
    expected_losses_inductor = np.array(
        [197.1, 250.8, 311.8, 381.3, 461.8, 548.6, 647.9, 758.7, 878.1, 1009.7]
    )
    assert problem.get_val("conduction_losses_inductor", units="W") == pytest.approx(
        expected_losses_inductor, rel=1e-2
    )
    expected_losses_capacitor = np.array(
        [49.1, 62.6, 77.8, 95.3, 115.4, 137.1, 161.9, 189.4, 219.1, 251.6]
    )
    assert problem.get_val("conduction_losses_capacitor", units="W") == pytest.approx(
        expected_losses_capacitor, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_switching_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesSwitchingLosses(
                dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    dc_current_out = np.linspace(200.0, 400.0, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesSwitchingLosses(
            dc_dc_converter_id="dc_dc_converter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    expected_losses_igbt = np.array(
        [131.3, 185.3, 244.7, 309.6, 380.4, 457.0, 539.7, 628.6, 724.0, 825.9]
    )
    assert problem.get_val("switching_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [73.6, 107.5, 146.0, 189.1, 236.8, 289.0, 345.8, 407.2, 473.0, 543.3]
    )
    assert problem.get_val("switching_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_total_losses_dc_dc_converter():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "switching_losses_IGBT",
        [131.3, 185.3, 244.7, 309.6, 380.4, 457.0, 539.7, 628.6, 724.0, 825.9],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [73.6, 107.5, 146.0, 189.1, 236.8, 289.0, 345.8, 407.2, 473.0, 543.3],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [148.9, 192.8, 243.1, 301.6, 371.3, 446.4, 534.6, 634.6, 743.0, 864.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [470.5, 552.7, 641.3, 737.0, 841.0, 950.7, 1069.4, 1196.4, 1330.4, 1472.9],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_inductor",
        [197.1, 250.8, 311.8, 381.3, 461.8, 548.6, 647.9, 758.7, 878.1, 1009.7],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_capacitor",
        [49.1, 62.6, 77.8, 95.3, 115.4, 137.1, 161.9, 189.4, 219.1, 251.6],
        units="W",
    )

    problem = run_system(PerformancesLosses(number_of_points=NB_POINTS_TEST), ivc)

    expected_losses = np.array(
        [1070.5, 1351.7, 1664.7, 2013.9, 2406.7, 2828.8, 3299.3, 3814.9, 4367.6, 4967.6]
    )
    assert problem.get_val("losses_converter", units="W") == pytest.approx(
        expected_losses, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_converter_efficiency():

    ivc = om.IndepVarComp()
    voltage_out = np.linspace(710, 910, NB_POINTS_TEST)
    ivc.add_output("dc_voltage_out", voltage_out, units="V")
    dc_current_out = np.linspace(200.0, 400.0, NB_POINTS_TEST)
    ivc.add_output("dc_current_out", dc_current_out, units="A")
    total_losses = np.array(
        [1070.5, 1351.7, 1664.7, 2013.9, 2406.7, 2828.8, 3299.3, 3814.9, 4367.6, 4967.6]
    )
    ivc.add_output("losses_converter", val=total_losses, units="W")

    # Won't be representative since the modulation index used for the computation of the losses
    # is not equal to the modulation index computed based on bus voltage
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_efficiency = np.array(
        [0.993, 0.992, 0.991, 0.99, 0.99, 0.989, 0.988, 0.988, 0.987, 0.987]
    )
    assert problem.get_val("efficiency") == pytest.approx(expected_efficiency, rel=1e-2)

    problem.check_partials(compact_print=True)
