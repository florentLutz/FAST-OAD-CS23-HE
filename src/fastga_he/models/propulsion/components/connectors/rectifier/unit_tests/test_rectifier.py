# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.perf_voltage_out_target import PerformancesVoltageOutTargetMission
from ..components.perf_switching_frequency import PerformancesSwitchingFrequencyMission
from ..components.perf_heat_sink_temperature import PerformancesHeatSinkTemperatureMission
from ..components.perf_switching_losses import PerformancesSwitchingLosses
from ..components.perf_resistance import PerformancesResistance
from ..components.perf_gate_voltage import PerformancesGateVoltage
from ..components.perf_conduction_loss import PerformancesConductionLosses
from ..components.perf_total_loss import PerformancesLosses
from ..components.perf_casing_temperature import PerformancesCasingTemperature
from ..components.perf_junction_temperature import PerformancesJunctionTemperature
from ..components.perf_junction_temperature_fixed import PerformancesJunctionTemperatureMission
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_efficiency_fixed import PerformancesEfficiencyMission
from ..components.perf_maximum import PerformancesMaximum

from ..components.sizing_energy_coefficient_scaling import SizingRectifierEnergyCoefficientScaling
from ..components.sizing_energy_coefficients import SizingRectifierEnergyCoefficients
from ..components.sizing_resistance_scaling import SizingRectifierResistanceScaling
from ..components.sizing_reference_resistance import SizingRectifierResistances
from ..components.sizing_thermal_resistance import SizingRectifierThermalResistances
from ..components.sizing_thermal_resistance_casing import SizingRectifierCasingThermalResistance
from ..components.sizing_capacitor_current_caliber import SizingRectifierCapacitorCurrentCaliber
from ..components.sizing_capacitor_capacity import SizingRectifierCapacitorCapacity
from ..components.sizing_dimension_module import SizingRectifierModuleDimension
from ..components.sizing_heat_sink_dimension import SizingRectifierHeatSinkDimension
from ..components.sizing_inductor_current_caliber import SizingRectifierInductorCurrentCaliber
from ..components.sizing_weight_casing import SizingRectifierCasingsWeight
from ..components.sizing_contactor_weight import SizingRectifierContactorWeight
from ..components.sizing_rectifier_weight import SizingRectifierWeight, SizingRectifierWeightBySum
from ..components.sizing_rectifier_cg_x import SizingRectifierCGX
from ..components.sizing_rectifier_cg_y import SizingRectifierCGY

from ..components.pre_lca_prod_weight_per_fu import PreLCARectifierProdWeightPerFU
from ..components.lcc_rectifier_cost import LCCRectifierCost

from ..components.sizing_rectifier import SizingRectifier
from ..components.perf_rectifier import PerformancesRectifier

from ..components.cstr_enforce import (
    ConstraintsCurrentRMS1PhaseEnforce,
    ConstraintsVoltagePeakEnforce,
    ConstraintsFrequencyEnforce,
    ConstraintsLossesEnforce,
)
from ..components.cstr_ensure import (
    ConstraintsCurrentRMS1PhaseEnsure,
    ConstraintsVoltagePeakEnsure,
    ConstraintsFrequencyEnsure,
    ConstraintsLossesEnsure,
)

from .....sub_components import SizingHeatSink, SizingInductor, SizingCapacitor

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..constants import POSSIBLE_POSITION

XML_FILE = "sample_rectifier.xml"
NB_POINTS_TEST = 10


def test_voltage_out_target_mission():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesVoltageOutTargetMission(
                rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PerformancesVoltageOutTargetMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("voltage_out_target", units="V") == pytest.approx(
        np.full(NB_POINTS_TEST, 800), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc2 = om.IndepVarComp()
    ivc2.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_out_target_mission",
        val=[850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 420.0],
        units="V",
    )

    problem2 = run_system(
        PerformancesVoltageOutTargetMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc2,
    )

    assert problem2.get_val("voltage_out_target", units="V") == pytest.approx(
        np.array([850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 850.0, 800.0, 700.0, 420.0]),
        rel=1e-2,
    )

    problem2.check_partials(compact_print=True)


def test_switching_frequency_mission():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:switching_frequency_mission",
        val=12.0e3,
        units="Hz",
    )

    problem = run_system(
        PerformancesSwitchingFrequencyMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("switching_frequency", units="Hz") == pytest.approx(
        np.full(NB_POINTS_TEST, 12.0e3), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:switching_frequency_mission",
        val=[15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 15e3, 12e3, 10e3, 420],
        units="Hz",
    )

    problem3 = run_system(
        PerformancesSwitchingFrequencyMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
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
        "data:propulsion:he_power_train:rectifier:rectifier_1:heat_sink_temperature_mission",
        val=290.0,
        units="degK",
    )

    problem = run_system(
        PerformancesHeatSinkTemperatureMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("heat_sink_temperature", units="degK") == pytest.approx(
        np.full(NB_POINTS_TEST, 290), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:heat_sink_temperature_mission",
        val=[290, 270, 290, 290, 270, 290, 290, 270, 290, 42],
        units="degK",
    )

    problem3 = run_system(
        PerformancesHeatSinkTemperatureMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc3,
    )

    assert problem3.get_val("heat_sink_temperature", units="degK") == pytest.approx(
        np.array([290, 270, 290, 290, 270, 290, 290, 270, 290, 42]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_switching_losses():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesSwitchingLosses(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "ac_current_rms_in_one_phase", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A"
    )
    ivc.add_output("switching_frequency", np.linspace(3000.0, 12000.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesSwitchingLosses(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )
    expected_losses_igbt = np.array(
        [107.1, 175.2, 263.6, 374.4, 510.0, 672.8, 865.1, 1089.3, 1347.7, 1642.6]
    )
    assert problem.get_val("switching_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array(
        [56.8, 82.6, 110.5, 139.4, 168.6, 197.3, 224.4, 249.2, 270.7, 288.2]
    )
    assert problem.get_val("switching_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_resistance_profile():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesResistance(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST)
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
        PerformancesResistance(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST), ivc
    )
    expected_resistance_igbt = (
        np.array([4.44, 4.69, 4.91, 5.11, 5.28, 5.43, 5.55, 5.65, 5.72, 5.77]) * 1e-3
    )
    assert problem.get_val("resistance_igbt", units="ohm") == pytest.approx(
        expected_resistance_igbt, rel=1e-2
    )
    resistance_diode = np.array([5.52, 5.77, 5.99, 6.19, 6.36, 6.51, 6.63, 6.73, 6.8, 6.85]) * 1e-3
    assert problem.get_val("resistance_diode", units="ohm") == pytest.approx(
        resistance_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_gate_voltage_profile():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesGateVoltage(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST)
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
        PerformancesGateVoltage(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST), ivc
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
            PerformancesConductionLosses(
                rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "ac_current_rms_in_one_phase", np.linspace(200.0, 500.0, NB_POINTS_TEST), units="A"
    )
    ivc.add_output("modulation_index", np.linspace(0.1, 1.0, NB_POINTS_TEST))
    ivc.add_output(
        "resistance_igbt",
        np.array([4.44, 4.69, 4.91, 5.11, 5.28, 5.43, 5.55, 5.65, 5.72, 5.77]) * 1e-3,
        units="ohm",
    )
    ivc.add_output(
        "resistance_diode",
        np.array([5.52, 5.77, 5.99, 6.19, 6.36, 6.51, 6.63, 6.73, 6.8, 6.85]) * 1e-3,
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
        PerformancesConductionLosses(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )
    expected_losses_igbt = np.array(
        [54.1, 74.4, 99.4, 129.8, 166.0, 208.6, 257.9, 314.2, 377.5, 448.3]
    )
    assert problem.get_val("conduction_losses_IGBT", units="W") == pytest.approx(
        expected_losses_igbt, rel=1e-2
    )
    expected_losses_diode = np.array([63.8, 72.5, 79.9, 85.6, 89.0, 89.5, 86.5, 79.6, 68.0, 51.3])
    assert problem.get_val("conduction_losses_diode", units="W") == pytest.approx(
        expected_losses_diode, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_total_losses_rectifier():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "switching_losses_IGBT",
        [107.1, 175.2, 263.6, 374.4, 510.0, 672.8, 865.1, 1089.3, 1347.7, 1642.6],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [56.8, 82.6, 110.5, 139.4, 168.6, 197.3, 224.4, 249.2, 270.7, 288.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [54.1, 74.4, 99.4, 129.8, 166.0, 208.6, 257.9, 314.2, 377.5, 448.3],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [63.8, 72.5, 79.9, 85.6, 89.0, 89.5, 86.5, 79.6, 68.0, 51.3],
        units="W",
    )

    problem = run_system(PerformancesLosses(number_of_points=NB_POINTS_TEST), ivc)

    expected_losses = np.array(
        [1690.8, 2428.2, 3320.4, 4375.2, 5601.6, 7009.2, 8603.4, 10393.8, 12383.4, 14582.4]
    )
    assert problem.get_val("losses_rectifier", units="W") == pytest.approx(
        expected_losses, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_perf_casing_temperature():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesCasingTemperature(
                rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    total_losses = np.array(
        [1690.8, 2428.2, 3320.4, 4375.2, 5601.6, 7009.2, 8603.4, 10393.8, 12383.4, 14582.4]
    )
    ivc.add_output("losses_rectifier", val=total_losses, units="W")
    ivc.add_output("heat_sink_temperature", units="degK", val=np.full_like(total_losses, 288.15))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCasingTemperature(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_temperature = np.array(
        [293.8, 296.2, 299.2, 302.7, 306.8, 311.5, 316.8, 322.8, 329.4, 336.8]
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
                rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "casing_temperature",
        units="degK",
        val=np.array([293.8, 296.2, 299.2, 302.7, 306.8, 311.5, 316.8, 322.8, 329.4, 336.8]),
    )
    ivc.add_output(
        "switching_losses_IGBT",
        [107.1, 175.2, 263.6, 374.4, 510.0, 672.8, 865.1, 1089.3, 1347.7, 1642.6],
        units="W",
    )
    ivc.add_output(
        "switching_losses_diode",
        [56.8, 82.6, 110.5, 139.4, 168.6, 197.3, 224.4, 249.2, 270.7, 288.2],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_IGBT",
        [54.1, 74.4, 99.4, 129.8, 166.0, 208.6, 257.9, 314.2, 377.5, 448.3],
        units="W",
    )
    ivc.add_output(
        "conduction_losses_diode",
        [63.8, 72.5, 79.9, 85.6, 89.0, 89.5, 86.5, 79.6, 68.0, 51.3],
        units="W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesJunctionTemperature(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    expected_temperature_diode = np.array(
        [332.5, 346.0, 360.3, 374.9, 389.5, 403.6, 416.6, 428.3, 438.1, 445.8]
    )
    assert problem.get_val(
        "diode_temperature",
        units="degK",
    ) == pytest.approx(expected_temperature_diode, rel=1e-2)
    expected_temperature_igbt = np.array(
        [333.9, 358.4, 389.6, 428.2, 475.1, 531.0, 596.4, 672.3, 759.0, 857.4]
    )
    assert problem.get_val(
        "IGBT_temperature",
        units="degK",
    ) == pytest.approx(expected_temperature_igbt, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_junction_temperature_constant():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:junction_temperature_mission",
        val=288.15,
        units="degK",
    )

    problem = run_system(
        PerformancesJunctionTemperatureMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
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
        "data:propulsion:he_power_train:rectifier:rectifier_1:junction_temperature_mission",
        val=[288.15, 298.15, 308.15, 318.15, 328.15, 338.15, 348.15, 358.15, 368.15, 378.15],
        units="degK",
    )

    problem3 = run_system(
        PerformancesJunctionTemperatureMission(
            rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST
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


def test_efficiency():
    # Will eventually disappear

    ivc = om.IndepVarComp()
    ivc.add_output(
        "losses_rectifier",
        val=np.array(
            [1690.8, 2428.2, 3320.4, 4375.2, 5601.6, 7009.2, 8603.4, 10393.8, 12383.4, 14582.4]
        ),
        units="W",
    )
    ivc.add_output("dc_current_out", units="A", val=np.linspace(300.0, 280.0, NB_POINTS_TEST))
    ivc.add_output("dc_voltage_out", units="V", val=np.linspace(500.0, 480.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesEfficiency(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    expected_efficiency = np.array(
        [0.989, 0.984, 0.978, 0.971, 0.962, 0.953, 0.942, 0.93, 0.917, 0.902]
    )
    assert problem.get_val("efficiency") == pytest.approx(expected_efficiency, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_efficiency_mission():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:efficiency_mission",
        val=0.98,
    )

    problem = run_system(
        PerformancesEfficiencyMission(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("efficiency") == pytest.approx(np.full(NB_POINTS_TEST, 0.98), rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc3 = om.IndepVarComp()
    ivc3.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:efficiency_mission",
        val=[0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9],
    )

    problem3 = run_system(
        PerformancesEfficiencyMission(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc3,
    )

    assert problem3.get_val("efficiency") == pytest.approx(
        np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9]),
        rel=1e-2,
    )

    problem3.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output("dc_current_out", units="A", val=np.linspace(300.0, 280.0, NB_POINTS_TEST))
    ivc.add_output("dc_voltage_out", units="V", val=np.linspace(500.0, 480.0, NB_POINTS_TEST))
    ivc.add_output("ac_voltage_peak_in", units="V", val=np.linspace(500.0, 480.0, NB_POINTS_TEST))
    ivc.add_output(
        "ac_current_rms_in_one_phase", units="A", val=np.linspace(133.0, 120.0, NB_POINTS_TEST)
    )
    ivc.add_output(
        "switching_frequency", units="Hz", val=np.linspace(10.0e3, 12.0e3, NB_POINTS_TEST)
    )
    ivc.add_output(
        "losses_rectifier",
        val=np.array(
            [1690.8, 2428.2, 3320.4, 4375.2, 5601.6, 7009.2, 8603.4, 10393.8, 12383.4, 14582.4]
        ),
        units="W",
    )

    problem = run_system(
        PerformancesMaximum(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:current_ac_max", units="A"
    ) == pytest.approx(133.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_ac_max", units="V"
    ) == pytest.approx(500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:current_dc_max", units="A"
    ) == pytest.approx(300.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_dc_max", units="V"
    ) == pytest.approx(500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:switching_frequency_max", units="Hz"
    ) == pytest.approx(12.0e3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:losses_max", units="W"
    ) == pytest.approx(14582.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_enforce_current():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentRMS1PhaseEnforce(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentRMS1PhaseEnforce(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:current_ac_caliber", units="A"
    ) == pytest.approx(133.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_enforce_voltage():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltagePeakEnforce(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsVoltagePeakEnforce(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_ac_caliber", units="V"
    ) == pytest.approx(800.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_enforce_frequency():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFrequencyEnforce(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsFrequencyEnforce(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:switching_frequency", units="Hz"
    ) == pytest.approx(12.0e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_enforce_losses():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsLossesEnforce(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsLossesEnforce(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:dissipable_heat",
        units="W",
    ) == pytest.approx(14582.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_ensure_current():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsCurrentRMS1PhaseEnsure(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsCurrentRMS1PhaseEnsure(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "constraints:propulsion:he_power_train:rectifier:rectifier_1:current_ac_caliber", units="A"
    ) == pytest.approx(-17.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_ensure_voltage():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltagePeakEnsure(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsVoltagePeakEnsure(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "constraints:propulsion:he_power_train:rectifier:rectifier_1:voltage_ac_caliber", units="V"
    ) == pytest.approx(-10.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_ensure_frequency():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFrequencyEnsure(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        ConstraintsFrequencyEnsure(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "constraints:propulsion:he_power_train:rectifier:rectifier_1:switching_frequency",
        units="Hz",
    ) == pytest.approx(-3.0e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_losses():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsLossesEnsure(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(ConstraintsLossesEnsure(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:rectifier:rectifier_1:dissipable_heat",
        units="W",
    ) == pytest.approx(-417.6, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_scaling_ratio():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierEnergyCoefficientScaling(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingRectifierEnergyCoefficientScaling(rectifier_id="rectifier_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:scaling:a"
    ) == pytest.approx(0.333, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:scaling:c"
    ) == pytest.approx(3.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_energy_coefficient():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierEnergyCoefficients(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierEnergyCoefficients(rectifier_id="rectifier_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_on:a"
    ) == pytest.approx(0.00528233, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_rr:a"
    ) == pytest.approx(0.00501952, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_off:a"
    ) == pytest.approx(0.00141464, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_on:b"
    ) == pytest.approx(3.326e-05, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_rr:b"
    ) == pytest.approx(0.000254, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_off:b"
    ) == pytest.approx(0.000340, rel=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_on:c"
    ) == pytest.approx(1.54040339e-06, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_rr:c"
    ) == pytest.approx(-5.21717365e-07, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:energy_off:c"
    ) == pytest.approx(-1.35349615e-07, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierResistanceScaling(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierResistanceScaling(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:scaling:resistance"
    ) == pytest.approx(3.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierResistances(rectifier_id="rectifier_1")), __file__, XML_FILE
    )

    problem = run_system(SizingRectifierResistances(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:igbt:resistance", units="ohm"
    ) == pytest.approx(4.53e-3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:diode:resistance", units="ohm"
    ) == pytest.approx(5.61e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_thermal_resistance_casing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierCasingThermalResistance(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierCasingThermalResistance(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:casing:thermal_resistance",
        units="K/W",
    ) == pytest.approx(0.010, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_thermal_resistance():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierThermalResistances(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierThermalResistances(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:igbt:thermal_resistance", units="K/W"
    ) == pytest.approx(0.249, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:diode:thermal_resistance", units="K/W"
    ) == pytest.approx(0.321, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_capacitor_current_caliber():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierCapacitorCurrentCaliber(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierCapacitorCurrentCaliber(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:capacitor:current_caliber",
        units="A",
    ) == pytest.approx(68.91, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_capacitor_capacity():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierCapacitorCapacity(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierCapacitorCapacity(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:capacitor:capacity",
        units="F",
    ) == pytest.approx(4.33e-4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_capacitor_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingCapacitor(prefix="data:propulsion:he_power_train:rectifier:rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingCapacitor(prefix="data:propulsion:he_power_train:rectifier:rectifier_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:capacitor:mass",
        units="kg",
    ) == pytest.approx(1.676, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_dimension_module():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierModuleDimension(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierModuleDimension(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:module:length", units="m"
    ) == pytest.approx(0.107, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:module:width", units="m"
    ) == pytest.approx(0.065, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:module:height", units="m"
    ) == pytest.approx(0.021, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_dimension():
    inputs_list = [
        "data:propulsion:he_power_train:rectifier:rectifier_1:module:length",
        "data:propulsion:he_power_train:rectifier:rectifier_1:module:width",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        inputs_list,
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierHeatSinkDimension(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:heat_sink:length", units="m"
    ) == pytest.approx(0.2145, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:heat_sink:width", units="m"
    ) == pytest.approx(0.1177, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inductor_current_caliber():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierInductorCurrentCaliber(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierInductorCurrentCaliber(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:inductor:current_caliber",
        units="A",
    ) == pytest.approx(150.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inductor():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingInductor(prefix="data:propulsion:he_power_train:rectifier:rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingInductor(prefix="data:propulsion:he_power_train:rectifier:rectifier_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:inductor:mass",
        units="kg",
    ) == pytest.approx(4.40, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight_casings():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierCasingsWeight(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierCasingsWeight(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:casing:mass", units="kg"
    ) == pytest.approx(0.705, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_contactor_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierContactorWeight(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierContactorWeight(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:contactor:mass",
        units="kg",
    ) == pytest.approx(2.32, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingHeatSink(prefix="data:propulsion:he_power_train:rectifier:rectifier_1")),
        __file__,
        XML_FILE,
    )
    ivc.add_output("data:propulsion:he_power_train:rectifier:rectifier_1:module:number", val=3)

    problem = run_system(
        SizingHeatSink(prefix="data:propulsion:he_power_train:rectifier:rectifier_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:heat_sink:mass",
        units="kg",
    ) == pytest.approx(0.444, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rectifier_weight():
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierWeight(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        SizingRectifierWeight(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:mass", units="kg"
    ) == pytest.approx(17.18, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rectifier_weight_by_sum():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifierWeightBySum(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifierWeightBySum(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:mass",
        units="kg",
    ) == pytest.approx(12.011, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:power_density",
        units="kW/kg",
    ) == pytest.approx(24.77, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_rectifier_cg_x():
    expected_cg = [2.69, 0.45, 2.54]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingRectifierCGX(rectifier_id="rectifier_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingRectifierCGX(rectifier_id="rectifier_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:rectifier:rectifier_1:CG:x",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_rectifier_cg_y():
    expected_cg = [2.34, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingRectifierCGY(rectifier_id="rectifier_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(SizingRectifierCGY(rectifier_id="rectifier_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:rectifier:rectifier_1:CG:y",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_sizing_rectifier():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingRectifier(rectifier_id="rectifier_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingRectifier(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:mass", units="kg"
    ) == pytest.approx(16.246, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:CG:y", units="m"
    ) == pytest.approx(2.34, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:low_speed:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)


def test_performances_rectifier():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRectifier(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("dc_voltage_out", units="V", val=np.linspace(500.0, 480.0, NB_POINTS_TEST))
    ivc.add_output("ac_voltage_peak_in", units="V", val=np.linspace(500.0, 480.0, NB_POINTS_TEST))
    ivc.add_output(
        "ac_voltage_rms_in",
        units="V",
        val=np.linspace(500.0, 480.0, NB_POINTS_TEST) / np.sqrt(3.0 / 2.0),
    )

    problem = run_system(
        PerformancesRectifier(rectifier_id="rectifier_1", number_of_points=NB_POINTS_TEST), ivc
    )

    om.n2(problem, show_browser=False)

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:rectifier:rectifier_1:mass",
        "data:environmental_impact:aircraft_per_fu",
        "data:TLAR:aircraft_lifespan",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCARectifierProdWeightPerFU(rectifier_id="rectifier_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:mass_per_fu", units="kg"
    ) == pytest.approx(3.436e-05, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:rectifier:rectifier_1:current_ac_max", units="A", val=133.0
    )

    problem = run_system(
        LCCRectifierCost(rectifier_id="rectifier_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:cost_per_unit", units="USD"
    ) == pytest.approx(2262.76, rel=1e-2)

    problem.check_partials(compact_print=True)
