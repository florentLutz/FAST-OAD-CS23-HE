# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import pytest

from ..components.sizing_energy_coefficient_scaling import SizingInverterEnergyCoefficientScaling
from ..components.sizing_energy_coefficients import SizingInverterEnergyCoefficients
from ..components.sizing_resistance_scaling import SizingInverterResistanceScaling
from ..components.sizing_reference_resistance import SizingInverterResistances
from ..components.sizing_thermal_resistance import SizingInverterThermalResistances
from ..components.sizing_thermal_resistance_casing import SizingInverterCasingThermalResistance
from ..components.sizing_weight_casing import SizingInverterCasingsWeight
from ..components.sizing_heat_capacity_casing import SizingInverterCasingHeatCapacity
from ..components.sizing_dimension_module import SizingInverterModuleDimension
from ..components.sizing_heat_sink_dimension import SizingInverterHeatSinkDimension
from ..components.sizing_capacitor_current_caliber import SizingInverterCapacitorCurrentCaliber
from ..components.sizing_capacitor_capacity import SizingInverterCapacitorCapacity
from ..components.sizing_inductor_inductance import SizingInverterInductorInductance
from ..components.sizing_inductor_current_caliber import SizingInverterInductorCurrentCaliber
from ..components.sizing_contactor_weight import SizingInverterContactorWeight
from ..components.sizing_inverter_weight import SizingInverterWeight
from ..components.sizing_inverter_power_density import SizingInverterPowerDensity
from ..components.sizing_inverter_cg_x import SizingInverterCGX
from ..components.sizing_inverter_cg_y import SizingInverterCGY
from ..components.sizing_inverter import SizingInverter
from ..components.perf_switching_frequency import PerformancesSwitchingFrequencyMission
from ..components.perf_heat_sink_temperature import PerformancesHeatSinkTemperatureMission
from ..components.perf_modulation_index import PerformancesModulationIndex
from ..components.perf_switching_losses import PerformancesSwitchingLosses
from ..components.perf_resistance import PerformancesResistance
from ..components.perf_gate_voltage import PerformancesGateVoltage
from ..components.perf_ac_power_out import PerformancesACPowerOut
from ..components.perf_conduction_loss import PerformancesConductionLosses
from ..components.perf_total_loss import PerformancesLosses
from fastga_he.models.propulsion.components.connectors.inverter.components.stale.perf_temperature_derivative import (
    PerformancesTemperatureDerivative,
)
from fastga_he.models.propulsion.components.connectors.inverter.components.stale.perf_temperature_increase import (
    PerformancesTemperatureIncrease,
)
from ..components.perf_casing_temperature import PerformancesCasingTemperature
from ..components.perf_junction_temperature import PerformancesJunctionTemperature
from ..components.perf_junction_temperature_fixed import PerformancesJunctionTemperatureMission
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_efficiency_fixed import PerformancesEfficiencyMission
from ..components.perf_dc_current import PerformancesDCCurrent
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_inverter import PerformancesInverter
from ..components.pre_lca_prod_weight_per_fu import PreLCAInverterProdWeightPerFU
from ..components.lcc_inverter_cost import LCCInverterCost
from ..components.lcc_inverter_operational_cost import LCCInverterOperationalCost

from ..components.cstr_enforce import (
    ConstraintsCurrentEnforce,
    ConstraintsVoltageEnforce,
    ConstraintsLossesEnforce,
    ConstraintsFrequencyEnforce,
)
from ..components.cstr_ensure import (
    ConstraintsCurrentEnsure,
    ConstraintsVoltageEnsure,
    ConstraintsLossesEnsure,
    ConstraintsFrequencyEnsure,
)

from .....sub_components import SizingHeatSink, SizingCapacitor, SizingInductor

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_inverter.xml"
NB_POINTS_TEST = 10


def test_scaling_ratio():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterEnergyCoefficientScaling(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingInverterEnergyCoefficientScaling(inverter_id="inverter_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:a"
    ) == pytest.approx(0.962, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:c"
    ) == pytest.approx(1.039, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_energy_coefficient():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterEnergyCoefficients(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterEnergyCoefficients(inverter_id="inverter_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:a"
    ) == pytest.approx(0.02197006, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:a"
    ) == pytest.approx(0.02087697, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:a"
    ) == pytest.approx(0.0058837, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:b"
    ) == pytest.approx(3.326e-05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:b"
    ) == pytest.approx(0.000254, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:b"
    ) == pytest.approx(0.000340, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_on:c"
    ) == pytest.approx(3.707e-7, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_rr:c"
    ) == pytest.approx(-1.256e-7, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:energy_off:c"
    ) == pytest.approx(-3.257e-8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterResistanceScaling(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterResistanceScaling(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:scaling:resistance"
    ) == pytest.approx(1.039, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterResistances(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterResistances(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:igbt:resistance", units="ohm"
    ) == pytest.approx(0.00209135, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:diode:resistance", units="ohm"
    ) == pytest.approx(0.00258995, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_thermal_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterThermalResistances(inverter_id="inverter_1")), __file__, XML_FILE
    )

    problem = run_system(SizingInverterThermalResistances(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:igbt:thermal_resistance", units="K/W"
    ) == pytest.approx(0.114955, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:diode:thermal_resistance", units="K/W"
    ) == pytest.approx(0.148195, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_thermal_resistance_casing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCasingThermalResistance(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCasingThermalResistance(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:casing:thermal_resistance", units="K/W"
    ) == pytest.approx(0.010, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_casings():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCasingsWeight(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCasingsWeight(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:casing:mass", units="kg"
    ) == pytest.approx(1.0446, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_capacity_casing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCasingHeatCapacity(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCasingHeatCapacity(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:casing:heat_capacity", units="J/degK"
    ) == pytest.approx(208.92, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_dimension_module():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterModuleDimension(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterModuleDimension(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:module:length", units="m"
    ) == pytest.approx(0.156, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:module:width", units="m"
    ) == pytest.approx(0.065, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:module:height", units="m"
    ) == pytest.approx(0.021, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_dimension():
    inputs_list = [
        "data:propulsion:he_power_train:inverter:inverter_1:module:length",
        "data:propulsion:he_power_train:inverter:inverter_1:module:width",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterHeatSinkDimension(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:length", units="m"
    ) == pytest.approx(0.2145, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:width", units="m"
    ) == pytest.approx(0.1716, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSink(prefix="data:propulsion:he_power_train:inverter:inverter_1")),
        __file__,
        XML_FILE,
    )
    ivc.add_output("data:propulsion:he_power_train:inverter:inverter_1:module:number", val=3)

    problem = run_system(
        SizingHeatSink(prefix="data:propulsion:he_power_train:inverter:inverter_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink:mass",
        units="kg",
    ) == pytest.approx(0.634, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_capacitor_current_caliber():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCapacitorCurrentCaliber(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCapacitorCurrentCaliber(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:capacitor:current_caliber",
        units="A",
    ) == pytest.approx(199.18, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_capacitor_capacity():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterCapacitorCapacity(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterCapacitorCapacity(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:capacitor:capacity",
        units="F",
    ) == pytest.approx(2.02e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_capacitor_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitor(prefix="data:propulsion:he_power_train:inverter:inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingCapacitor(prefix="data:propulsion:he_power_train:inverter:inverter_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:capacitor:mass",
        units="kg",
    ) == pytest.approx(7.82, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inductor_inductance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterInductorInductance(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterInductorInductance(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:inductor:inductance",
        units="H",
    ) == pytest.approx(11.5e-6, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inductor_current_caliber():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterInductorCurrentCaliber(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterInductorCurrentCaliber(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:inductor:current_caliber",
        units="A",
    ) == pytest.approx(430.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inductor_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductor(prefix="data:propulsion:he_power_train:inverter:inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingInductor(prefix="data:propulsion:he_power_train:inverter:inverter_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:inductor:mass",
        units="kg",
    ) == pytest.approx(4.98, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_contactor_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterContactorWeight(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterContactorWeight(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:contactor:mass",
        units="kg",
    ) == pytest.approx(4.85, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inverter_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterWeight(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterWeight(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:mass",
        units="kg",
    ) == pytest.approx(26.3456, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inverter_power_density():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverterPowerDensity(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverterPowerDensity(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:power_density",
        units="kW/kg",
    ) == pytest.approx(15.17, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inverter_cg_x():
    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingInverterCGX(inverter_id="inverter_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingInverterCGX(inverter_id="inverter_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:inverter:inverter_1:CG:x",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_inverter_cg_y():
    expected_cg = [2.5, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingInverterCGY(inverter_id="inverter_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingInverterCGY(inverter_id="inverter_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:inverter:inverter_1:CG:y",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_inverter_sizing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInverter(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingInverter(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:mass",
        units="kg",
    ) == pytest.approx(28.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:power_density",
        units="kW/kg",
    ) == pytest.approx(20.9, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:cruise:CD0",
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:CG:y", units="m"
    ) == pytest.approx(2.5, rel=1e-2)


def test_constraints_enforce_current():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnforce(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsCurrentEnforce(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:current_caliber",
        units="A",
    ) == pytest.approx(400.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_enforce_voltage():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsVoltageEnforce(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:voltage_caliber",
        units="V",
    ) == pytest.approx(500.00, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_enforce_losses():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsLossesEnforce(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsLossesEnforce(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:dissipable_heat",
        units="W",
    ) == pytest.approx(11000.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_enforce_frequency():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFrequencyEnforce(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsFrequencyEnforce(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:switching_frequency",
        units="Hz",
    ) == pytest.approx(12.0e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_current():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentEnsure(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsCurrentEnsure(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:inverter:inverter_1:current_caliber",
        units="A",
    ) == pytest.approx(-33.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_voltage():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsVoltageEnsure(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:inverter:inverter_1:voltage_caliber",
        units="V",
    ) == pytest.approx(0.00, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_losses():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsLossesEnsure(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsLossesEnsure(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:inverter:inverter_1:dissipable_heat",
        units="W",
    ) == pytest.approx(-808.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_frequency():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFrequencyEnsure(inverter_id="inverter_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsFrequencyEnsure(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:inverter:inverter_1:switching_frequency",
        units="Hz",
    ) == pytest.approx(-3.0e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_switching_frequency_mission():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:switching_frequency_mission",
        val=12.0e3,
        units="Hz",
    )

    problem = run_system(
        PerformancesSwitchingFrequencyMission(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("switching_frequency", units="Hz") == pytest.approx(
        np.full(NB_POINTS_TEST, 12.0e3), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:switching_frequency_mission",
        val=[15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 420],
        units="Hz",
    )

    problem3 = run_system(
        PerformancesSwitchingFrequencyMission(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc3,
    )

    assert problem3.get_val("switching_frequency", units="Hz") == pytest.approx(
        np.array([15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 420]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_heat_sink_temperature_mission():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink_temperature_mission",
        val=290.0,
        units="degK",
    )

    problem = run_system(
        PerformancesHeatSinkTemperatureMission(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("heat_sink_temperature", units="degK") == pytest.approx(
        np.full(NB_POINTS_TEST, 290), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:heat_sink_temperature_mission",
        val=[290, 270, 290, 290, 270, 290, 290, 270, 290, 42],
        units="degK",
    )

    problem3 = run_system(
        PerformancesHeatSinkTemperatureMission(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc3,
    )

    assert problem3.get_val("heat_sink_temperature", units="degK") == pytest.approx(
        np.array([290, 270, 290, 290, 270, 290, 290, 270, 290, 42]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_modulation_idx():
    ivc = om.IndepVarComp()
    ivc.add_output("dc_voltage_in", units="V", val=np.full(NB_POINTS_TEST, 1000.0))
    ivc.add_output(
        "ac_voltage_peak_out",
        units="V",
        val=np.array([710.4, 728.5, 747.6, 767.2, 787.1, 807.5, 827.9, 848.6, 869.6, 890.5]),
    )

    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem(
        name="ivc",
        subsys=ivc,
        promotes=["*"],
    )
    model.add_subsystem(
        name="modulation_idx",
        subsys=PerformancesModulationIndex(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.nonlinear_solver.options["iprint"] = 0
    model.nonlinear_solver.options["maxiter"] = 200
    model.nonlinear_solver.options["rtol"] = 1e-5
    model.linear_solver = om.DirectSolver()

    problem.setup()
    problem.run_model()

    assert problem.get_val("modulation_index") == pytest.approx(
        np.array([0.71, 0.73, 0.75, 0.77, 0.79, 0.81, 0.83, 0.85, 0.87, 0.89]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_switching_losses():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesSwitchingLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "ac_current_rms_out_one_phase", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A"
    )
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesSwitchingLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_losses_igbt = np.array(
        [126.5, 184.4, 250.8, 326.2, 411.0, 505.5, 610.2, 725.5, 851.8, 989.5]
    )
    assert problem.get_val("switching_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [72.8, 111.0, 156.1, 208.1, 266.8, 332.2, 404.4, 483.1, 568.4, 660.2]
    )
    assert problem.get_val("switching_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_resistance_profile():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesResistance(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    temperature = np.array(
        [288.15, 301.626, 313.719, 324.429, 333.762, 341.721, 348.306, 353.523, 357.375, 359.862]
    )
    ivc.add_output("diode_temperature", units="degK", val=temperature)
    ivc.add_output("IGBT_temperature", units="degK", val=temperature)

    problem = run_system(
        PerformancesResistance(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_resistance_igbt = (
        np.array([2.05, 2.16, 2.27, 2.36, 2.44, 2.51, 2.56, 2.61, 2.64, 2.66]) * 1e-3
    )
    assert problem.get_val("resistance_igbt", units="ohm") == pytest.approx(
        expected_resistance_igbt, rel=1e-2
    )
    resistance_diode = np.array([2.55, 2.66, 2.77, 2.86, 2.94, 3.01, 3.06, 3.11, 3.14, 3.16]) * 1e-3
    assert problem.get_val("resistance_diode", units="ohm") == pytest.approx(
        resistance_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_gate_voltage_profile():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesGateVoltage(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    temperature = np.array(
        [288.15, 301.626, 313.719, 324.429, 333.762, 341.721, 348.306, 353.523, 357.375, 359.862]
    )
    ivc.add_output("diode_temperature", units="degK", val=temperature)
    ivc.add_output("IGBT_temperature", units="degK", val=temperature)

    problem = run_system(
        PerformancesGateVoltage(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_gate_voltage_igbt = np.array(
        [0.875, 0.862, 0.851, 0.841, 0.833, 0.826, 0.82, 0.815, 0.811, 0.809]
    )
    assert problem.get_val("gate_voltage_igbt", units="V") == pytest.approx(
        expected_gate_voltage_igbt, rel=1e-2
    )
    expected_gate_voltage_diode = np.array(
        [1.314, 1.276, 1.241, 1.211, 1.184, 1.161, 1.142, 1.127, 1.116, 1.109]
    )
    assert problem.get_val("gate_voltage_diode", units="V") == pytest.approx(
        expected_gate_voltage_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_conduction_losses():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesConductionLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "ac_current_rms_out_one_phase", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A"
    )
    ivc.add_output("modulation_index", np.linspace(0.1, 1.0, NB_POINTS_TEST))
    ivc.add_output(
        "resistance_igbt",
        np.array([2.05, 2.16, 2.27, 2.36, 2.44, 2.51, 2.56, 2.61, 2.64, 2.66]) * 1e-3,
        units="ohm",
    )
    ivc.add_output(
        "resistance_diode",
        np.array([2.55, 2.66, 2.77, 2.86, 2.94, 3.01, 3.06, 3.11, 3.14, 3.16]) * 1e-3,
        units="ohm",
    )
    ivc.add_output(
        "gate_voltage_igbt",
        np.array([0.875, 0.862, 0.851, 0.841, 0.833, 0.826, 0.82, 0.815, 0.811, 0.809]),
        units="V",
    )
    ivc.add_output(
        "gate_voltage_diode",
        np.array([1.314, 1.276, 1.241, 1.211, 1.184, 1.161, 1.142, 1.127, 1.116, 1.109]),
        units="V",
    )

    problem = run_system(
        PerformancesConductionLosses(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_losses_igbt = np.array(
        [41.2, 54.2, 69.9, 88.3, 109.8, 134.6, 162.5, 194.4, 229.6, 268.6]
    )
    assert problem.get_val("conduction_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array([50.2, 55.0, 58.6, 60.9, 61.6, 60.6, 57.6, 52.3, 44.5, 33.9])
    assert problem.get_val("conduction_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_total_losses_inverter():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "switching_losses_IGBT",
        [126.5, 184.4, 250.8, 326.2, 411.0, 505.5, 610.2, 725.5, 851.8, 989.5],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [72.8, 111.0, 156.1, 208.1, 266.8, 332.2, 404.4, 483.1, 568.4, 660.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [40.99, 54.58, 70.94, 90.15, 112.55, 138.36, 167.46, 200.56, 237.06, 277.29],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [49.8, 55.72, 60.53, 63.82, 65.39, 64.94, 62.1, 56.75, 48.48, 37.13],
        units="W",
    )

    problem = run_system(PerformancesLosses(number_of_points=NB_POINTS_TEST), ivc)

    expected_losses = np.array(
        [1740.54, 2434.2, 3230.22, 4129.62, 5134.44, 6246.0, 7464.96, 8795.46, 10234.44, 11784.72]
    )
    assert problem.get_val("losses_inverter", units="W") == pytest.approx(expected_losses, rel=1e-2)

    problem.check_partials(compact_print=True)


def _test_temperature_derivative():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTemperatureDerivative(
                inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    total_losses = np.array(
        [1740.54, 2434.2, 3230.22, 4129.62, 5134.44, 6246.0, 7464.96, 8795.46, 10234.44, 11784.72]
    )
    ivc.add_output("losses_inverter", val=total_losses, units="W")
    temperature = np.array(
        [288.15, 301.626, 313.719, 324.429, 333.762, 341.721, 348.306, 353.523, 357.375, 359.862]
    )
    ivc.add_output("inverter_temperature", units="degK", val=temperature)
    ivc.add_output("heat_sink_temperature", units="degK", val=np.full_like(temperature, 288.15))

    problem = run_system(
        PerformancesTemperatureDerivative(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("inverter_temperature_time_derivative", units="degK/s") == pytest.approx(
        np.array([2.78, 0.81, -0.67, -1.68, -2.2, -2.24, -1.8, -0.87, 0.55, 2.46]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def _test_perf_temperature_increase():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesTemperatureIncrease(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    derivative = np.array([2.78, 0.81, -0.67, -1.68, -2.2, -2.24, -1.8, -0.87, 0.55, 2.46])
    ivc.add_output("inverter_temperature_time_derivative", units="degK/s", val=derivative)
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 300.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTemperatureIncrease(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_increase = np.array(
        [834.0, 243.0, -201.0, -504.0, -660.0, -672.0, -540.0, -261.0, 165.0, 738.0]
    )
    assert problem.get_val(
        "inverter_temperature_increase",
        units="degK",
    ) == pytest.approx(expected_increase, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_casing_temperature():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesCasingTemperature(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    total_losses = np.array(
        [1740.54, 2434.2, 3230.22, 4129.62, 5134.44, 6246.0, 7464.96, 8795.46, 10234.44, 11784.72]
    )
    ivc.add_output("losses_inverter", val=total_losses, units="W")
    ivc.add_output("heat_sink_temperature", units="degK", val=np.full_like(total_losses, 288.15))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCasingTemperature(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_temperature = np.array(
        [293.95, 296.26, 298.92, 301.92, 305.26, 308.97, 313.03, 317.47, 322.26, 327.43]
    )
    assert problem.get_val(
        "casing_temperature",
        units="degK",
    ) == pytest.approx(expected_temperature, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_junction_temperature():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesJunctionTemperature(
                inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "casing_temperature",
        units="degK",
        val=np.array(
            [293.95, 296.26, 298.92, 301.92, 305.26, 308.97, 313.03, 317.47, 322.26, 327.43]
        ),
    )
    ivc.add_output(
        "switching_losses_IGBT",
        [126.5, 184.4, 250.8, 326.2, 411.0, 505.5, 610.2, 725.5, 851.8, 989.5],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [72.8, 111.0, 156.1, 208.1, 266.8, 332.2, 404.4, 483.1, 568.4, 660.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [40.99, 54.58, 70.94, 90.15, 112.55, 138.36, 167.46, 200.56, 237.06, 277.29],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [49.8, 55.72, 60.53, 63.82, 65.39, 64.94, 62.1, 56.75, 48.48, 37.13],
        units="W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesJunctionTemperature(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_temperature_diode = np.array(
        [312.12, 320.97, 331.02, 342.22, 354.49, 367.82, 382.16, 397.47, 413.68, 430.77]
    )
    assert problem.get_val(
        "diode_temperature",
        units="degK",
    ) == pytest.approx(expected_temperature_diode, rel=1e-2)
    expected_temperature_igbt = np.array(
        [313.2, 323.73, 335.91, 349.78, 365.44, 382.98, 402.43, 423.93, 447.43, 473.05]
    )
    assert problem.get_val(
        "IGBT_temperature",
        units="degK",
    ) == pytest.approx(expected_temperature_igbt, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_junction_temperature_constant():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:junction_temperature_mission",
        val=288.15,
        units="degK",
    )

    problem = run_system(
        PerformancesJunctionTemperatureMission(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("diode_temperature", units="degK") == pytest.approx(
        np.full(NB_POINTS_TEST, 288.15), rel=1e-2
    )
    assert problem.get_val("IGBT_temperature", units="degK") == pytest.approx(
        np.full(NB_POINTS_TEST, 288.15), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:junction_temperature_mission",
        val=[288.15, 298.15, 308.15, 318.15, 328.15, 338.15, 348.15, 358.15, 368.15, 378.15],
        units="degK",
    )

    problem3 = run_system(
        PerformancesJunctionTemperatureMission(
            inverter_id="inverter_1", number_of_points=NB_POINTS_TEST
        ),
        ivc3,
    )

    assert problem3.get_val("diode_temperature", units="degK") == pytest.approx(
        np.array([288.15, 298.15, 308.15, 318.15, 328.15, 338.15, 348.15, 358.15, 368.15, 378.15]),
        rel=1e-2,
    )
    assert problem3.get_val("IGBT_temperature", units="degK") == pytest.approx(
        np.array([288.15, 298.15, 308.15, 318.15, 328.15, 338.15, 348.15, 358.15, 368.15, 378.15]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_inverter_efficiency():
    ivc = om.IndepVarComp()
    ivc.add_output(
        name="ac_voltage_rms_out",
        val=np.array([580.0, 594.8, 610.4, 626.4, 642.7, 659.3, 676.0, 692.9, 710.0, 727.1]),
        units="V",
    )
    ivc.add_output(
        name="ac_current_rms_out_one_phase",
        val=np.linspace(200.0, 500.0, NB_POINTS_TEST),
        units="A",
    )
    total_losses = np.array(
        [1740.54, 2434.2, 3230.22, 4129.62, 5134.44, 6246.0, 7464.96, 8795.46, 10234.44, 11784.72]
    )
    ivc.add_output("losses_inverter", val=total_losses, units="W")

    # Won't be representative since the modulation index used for the computation of the losses
    # is not equal to the modulation index computed based on bus voltage
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_efficiency = np.array(
        [0.995, 0.994, 0.993, 0.993, 0.992, 0.991, 0.991, 0.99, 0.99, 0.989]
    )
    assert problem.get_val("efficiency") == pytest.approx(expected_efficiency, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_efficiency_mission():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:efficiency_mission",
        val=0.98,
    )

    problem = run_system(
        PerformancesEfficiencyMission(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("efficiency") == pytest.approx(np.full(NB_POINTS_TEST, 0.98), rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:inverter:inverter_1:efficiency_mission",
        val=[0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9],
    )

    problem3 = run_system(
        PerformancesEfficiencyMission(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST),
        ivc3,
    )

    assert problem3.get_val("efficiency") == pytest.approx(
        np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_ac_power_out():
    ivc = om.IndepVarComp()
    ivc.add_output(
        name="ac_voltage_rms_out",
        val=np.array([580.0, 594.8, 610.4, 626.4, 642.7, 659.3, 676.0, 692.9, 710.0, 727.1]),
        units="V",
    )
    ivc.add_output(
        name="ac_current_rms_out_one_phase",
        val=np.linspace(200.0, 500.0, NB_POINTS_TEST),
        units="A",
    )

    problem = run_system(
        PerformancesACPowerOut(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    power_rating = np.array(
        [348.0, 416.36, 488.32, 563.76, 642.7, 725.23, 811.2, 900.77, 994.0, 1090.65]
    )
    assert problem.get_val("ac_power_out", units="kW") == pytest.approx(power_rating, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_dc_current():
    ivc = om.IndepVarComp()
    ivc.add_output("dc_voltage_in", units="V", val=np.full(NB_POINTS_TEST, 1000.0))
    ivc.add_output(
        "efficiency",
        val=np.array([0.995, 0.994, 0.993, 0.993, 0.992, 0.991, 0.991, 0.99, 0.99, 0.989]),
    )
    ivc.add_output(
        name="ac_voltage_rms_out",
        val=np.array([580.0, 594.8, 610.4, 626.4, 642.7, 659.3, 676.0, 692.9, 710.0, 727.1]),
        units="V",
    )
    ivc.add_output(
        name="ac_current_rms_out_one_phase",
        val=np.linspace(200.0, 500.0, NB_POINTS_TEST),
        units="A",
    )

    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem(
        name="ivc",
        subsys=ivc,
        promotes=["*"],
    )
    model.add_subsystem(
        name="dc_current",
        subsys=PerformancesDCCurrent(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.nonlinear_solver.options["iprint"] = 0
    model.nonlinear_solver.options["maxiter"] = 200
    model.nonlinear_solver.options["rtol"] = 1e-5
    model.linear_solver = om.DirectSolver()

    problem.setup()
    problem.run_model()

    assert problem.get_val("dc_current_in", units="A") == pytest.approx(
        np.array(
            [349.75, 418.87, 491.76, 567.73, 647.88, 731.82, 818.57, 909.87, 1004.04, 1102.78]
        ),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output("dc_voltage_in", units="V", val=np.full(NB_POINTS_TEST, 1000.0))
    ivc.add_output(
        "dc_current_in",
        units="A",
        val=np.array(
            [349.75, 418.87, 491.76, 567.73, 647.88, 731.82, 818.57, 909.87, 1004.04, 1102.78]
        ),
    )
    ivc.add_output(
        name="ac_current_rms_out_one_phase",
        val=np.linspace(200.0, 500.0, NB_POINTS_TEST),
        units="A",
    )
    ivc.add_output(
        "ac_voltage_peak_out",
        units="V",
        val=np.array([710.4, 728.5, 747.6, 767.2, 787.1, 807.5, 827.9, 848.6, 869.6, 890.5]),
    )
    ivc.add_output(
        "ac_power_out",
        units="kW",
        val=np.array([348.0, 416.36, 488.32, 563.76, 642.7, 725.23, 811.2, 900.77, 994.0, 1090.65]),
    )
    ivc.add_output(
        "diode_temperature",
        units="degK",
        val=np.array(
            [312.12, 320.97, 331.02, 342.22, 354.49, 367.82, 382.16, 397.47, 413.68, 430.77]
        ),
    )
    ivc.add_output(
        "IGBT_temperature",
        units="degK",
        val=np.array(
            [313.2, 323.73, 335.91, 349.78, 365.44, 382.98, 402.43, 423.93, 447.43, 473.05]
        ),
    )
    ivc.add_output(
        "casing_temperature",
        units="degK",
        val=np.array(
            [293.95, 296.26, 298.92, 301.92, 305.26, 308.97, 313.03, 317.47, 322.26, 327.43]
        ),
    )
    total_losses = np.array(
        [1740.54, 2434.2, 3230.22, 4129.62, 5134.44, 6246.0, 7464.96, 8795.46, 10234.44, 11784.72]
    )
    ivc.add_output("losses_inverter", val=total_losses, units="W")
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))
    ivc.add_output("modulation_index", np.linspace(0.1, 1.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesMaximum(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:current_ac_max", units="A"
    ) == pytest.approx(500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:voltage_ac_max", units="V"
    ) == pytest.approx(890.5, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:current_dc_max", units="A"
    ) == pytest.approx(1102.78, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:voltage_dc_max", units="V"
    ) == pytest.approx(1000.5, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:ac_power_out_max", units="kW"
    ) == pytest.approx(1090.65, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:igbt:temperature_max", units="degK"
    ) == pytest.approx(473.05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:diode:temperature_max", units="degK"
    ) == pytest.approx(430.77, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:casing:temperature_max", units="degK"
    ) == pytest.approx(327.43, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:losses_max", units="W"
    ) == pytest.approx(11784.72, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:switching_frequency_max", units="Hz"
    ) == pytest.approx(12.0e3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:modulation_idx_max"
    ) == pytest.approx(1.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_inverter_tot():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesInverter(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "ac_current_rms_out_one_phase", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A"
    )
    ivc.add_output("dc_voltage_in", units="V", val=np.full(NB_POINTS_TEST, 1000.0))
    ivc.add_output(
        "ac_voltage_peak_out",
        units="V",
        val=np.array([710.4, 728.5, 747.6, 767.2, 787.1, 807.5, 827.9, 848.6, 869.6, 890.5]),
    )
    ivc.add_output(
        name="ac_voltage_rms_out",
        val=np.array([580.0, 594.8, 610.4, 626.4, 642.7, 659.3, 676.0, 692.9, 710.0, 727.1]),
        units="V",
    )

    problem = run_system(
        PerformancesInverter(inverter_id="inverter_1", number_of_points=NB_POINTS_TEST),
        ivc,
        add_solvers=True,
    )

    expected_efficiency = np.array(
        [0.985, 0.986, 0.987, 0.987, 0.988, 0.988, 0.988, 0.989, 0.989, 0.989]
    )
    assert problem.get_val("efficiency") == pytest.approx(expected_efficiency, rel=1e-2)
    expected_temperature_igbt = np.array(
        [371.6, 381.1, 391.2, 401.8, 413.1, 424.9, 437.4, 450.6, 464.5, 479.3]
    )
    assert problem.get_val("IGBT_temperature", units="degK") == pytest.approx(
        expected_temperature_igbt, rel=1e-2
    )
    expected_temperature_diode = np.array(
        [352.7, 361.7, 370.8, 380.0, 389.2, 398.4, 407.6, 416.9, 426.2, 435.5]
    )
    assert problem.get_val("diode_temperature", units="degK") == pytest.approx(
        expected_temperature_diode, rel=1e-2
    )
    expected_dc_current_in = np.array(
        [353.3, 422.31, 494.94, 571.07, 650.71, 733.97, 820.68, 911.02, 1005.04, 1102.5]
    )
    assert problem.get_val("dc_current_in", units="A") == pytest.approx(
        expected_dc_current_in, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:inverter:inverter_1:mass",
        "data:environmental_impact:aircraft_per_fu",
        "data:TLAR:aircraft_lifespan",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAInverterProdWeightPerFU(inverter_id="inverter_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:mass_per_fu", units="kg"
    ) == pytest.approx(8.564e-05, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        name="data:propulsion:he_power_train:inverter:inverter_1:ac_power_out_max",
        val=200.0,
        units="kW",
    )

    problem = run_system(
        LCCInverterCost(inverter_id="inverter_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:purchase_cost", units="USD"
    ) == pytest.approx(7629.22, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_operational_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        name="data:propulsion:he_power_train:inverter:inverter_1:purchase_cost",
        val=18391.45,
        units="USD",
    )

    problem = run_system(
        LCCInverterOperationalCost(inverter_id="inverter_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:operational_cost", units="USD/yr"
    ) == pytest.approx(1226.096, rel=1e-3)

    problem.check_partials(compact_print=True)
