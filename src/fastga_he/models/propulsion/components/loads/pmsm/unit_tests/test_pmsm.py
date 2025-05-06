# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.sizing_diameter_scaling import SizingMotorDiameterScaling
from ..components.sizing_diameter import SizingMotorDiameter
from ..components.sizing_length_scaling import SizingMotorLengthScaling
from ..components.sizing_length import SizingMotorLength
from ..components.sizing_weight import SizingMotorWeight
from ..components.sizing_resistance_scaling import SizingMotorPhaseResistanceScaling
from ..components.sizing_resistance import SizingMotorPhaseResistance
from ..components.sizing_torque_constant_scaling import SizingMotorTorqueConstantScaling
from ..components.sizing_torque_constant import SizingMotorTorqueConstant
from ..components.sizing_loss_coefficient_scaling import SizingMotorLossCoefficientScaling
from ..components.sizing_loss_coefficient import SizingMotorLossCoefficient
from ..components.sizing_pmsm_cg_x import SizingPMSMCGX
from ..components.sizing_pmsm_cg_y import SizingPMSMCGY
from ..components.sizing_pmsm_drag import SizingPMSMDrag
from ..components.perf_torque import PerformancesTorque
from ..components.perf_losses import PerformancesLosses
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_current_rms import PerformancesCurrentRMS
from ..components.perf_current_rms_phase import PerformancesCurrentRMS1Phase
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum
from ..components.pre_lca_prod_weight_per_fu import PreLCAMotorProdWeightPerFU
from ..components.lcc_pmsm_cost import LCCPMSMCost
from ..components.lcc_pmsm_operational_cost import LCCPMSMOperationalCost

from ..components.cstr_enforce import (
    ConstraintsTorqueEnforce,
    ConstraintsRPMEnforce,
    ConstraintsVoltageEnforce,
)
from ..components.cstr_ensure import (
    ConstraintsTorqueEnsure,
    ConstraintsRPMEnsure,
    ConstraintsVoltageEnsure,
)
from ..components.cstr_pmsm import ConstraintPMSMPowerRateMission

from ..components.sizing_pmsm import SizingPMSM
from ..components.perf_pmsm import PerformancesPMSM

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_motor.xml"
NB_POINTS_TEST = 10


def test_diameter_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorDiameterScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorDiameterScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:scaling:diameter"
    ) == pytest.approx(0.9, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diameter():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorDiameter(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorDiameter(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:diameter", units="m"
    ) == pytest.approx(0.241, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorLengthScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLengthScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:scaling:length"
    ) == pytest.approx(0.97, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length():
    ivc = get_indep_var_comp(list_inputs(SizingMotorLength(motor_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLength(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:length", units="m"
    ) == pytest.approx(0.088, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight():
    ivc = get_indep_var_comp(list_inputs(SizingMotorWeight(motor_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorWeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(16.19, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorPhaseResistanceScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorPhaseResistanceScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:scaling:phase_resistance"
    ) == pytest.approx(0.93, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorPhaseResistance(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorPhaseResistance(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:phase_resistance", units="ohm"
    ) == pytest.approx(21.36e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque_constant_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorTorqueConstantScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorTorqueConstantScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:scaling:torque_constant"
    ) == pytest.approx(0.77, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque_constant():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorTorqueConstant(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorTorqueConstant(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:torque_constant", units="N*m/A"
    ) == pytest.approx(1.46, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_loss_coefficient_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorLossCoefficientScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLossCoefficientScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:scaling:alpha"
    ) == pytest.approx(1.57, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:scaling:beta"
    ) == pytest.approx(0.51, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:scaling:gamma"
    ) == pytest.approx(0.55, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_loss_coefficient():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorLossCoefficient(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLossCoefficient(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:loss_coefficient:alpha", units="W/N**2/m**2"
    ) == pytest.approx(0.025, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:loss_coefficient:beta", units="W*s/rad"
    ) == pytest.approx(3.38, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:loss_coefficient:gamma", units="W*s**2/rad**2"
    ) == pytest.approx(0.00825, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_motor_cg_x():
    expected_cg = [2.39, 0.25]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingPMSMCGX(motor_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingPMSMCGX(motor_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:PMSM:motor_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_cg_y():
    expected_cg = [1.5, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingPMSMCGY(motor_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingPMSMCGY(motor_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:PMSM:motor_1:CG:y", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_drag():
    expected_drag_ls = [0.357, 0.0]
    expected_drag_cruise = [0.352, 0.0]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_drag_ls, expected_drag_cruise
    ):
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingPMSMDrag(motor_id="motor_1", position=option, low_speed_aero=ls_option)
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingPMSMDrag(motor_id="motor_1", position=option, low_speed_aero=ls_option), ivc
            )

            if ls_option:
                assert problem.get_val(
                    "data:propulsion:he_power_train:PMSM:motor_1:low_speed:CD0",
                ) * 1e3 == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:PMSM:motor_1:cruise:CD0",
                ) * 1e3 == pytest.approx(cruise_drag, rel=1e-2)

            # Slight error on reynolds is due to step
            problem.check_partials(compact_print=True)


def test_constraints_torque_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnforce(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(150.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnforce(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:voltage_caliber", units="V"
    ) == pytest.approx(156.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_torque_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnsure(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(-500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:PMSM:motor_1:voltage_caliber", units="V"
    ) == pytest.approx(-543.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_power_for_power_rate():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintPMSMPowerRateMission(motor_id="motor_1")),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintPMSMPowerRateMission(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:shaft_power_rating", units="kW"
    ) == pytest.approx(70.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(PerformancesTorque(number_of_points=NB_POINTS_TEST)), __file__, XML_FILE
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm", np.linspace(3500, 4500, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_out", units="N*m") == pytest.approx(
        [82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0], rel=1e-2
    )
    assert problem.get_val("shaft_power_for_power_rate", units="kW") == pytest.approx(
        np.linspace(30, 70, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_losses():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "torque_out",
        np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
        units="N*m",
    )
    ivc.add_output("rpm", np.linspace(3500, 4500, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("power_losses", units="kW") == pytest.approx(
        [2.52, 2.66, 2.82, 2.98, 3.14, 3.3, 3.47, 3.63, 3.81, 3.98], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_efficiency():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesEfficiency(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, NB_POINTS_TEST), units="kW")
    ivc.add_output(
        "power_losses",
        np.array([2.52, 2.66, 2.82, 2.98, 3.14, 3.3, 3.47, 3.63, 3.81, 3.98]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesEfficiency(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("efficiency") == pytest.approx(
        [0.923, 0.928, 0.932, 0.936, 0.938, 0.941, 0.942, 0.944, 0.945, 0.946], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_active_power():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesActivePower(number_of_points=NB_POINTS_TEST)), __file__, XML_FILE
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, NB_POINTS_TEST), units="kW")
    ivc.add_output(
        "efficiency",
        np.array([0.923, 0.928, 0.932, 0.936, 0.938, 0.941, 0.942, 0.944, 0.945, 0.946]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesActivePower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("active_power", units="kW") == pytest.approx(
        [32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_apparent_power():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesApparentPower(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "active_power",
        np.array([32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesApparentPower(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("apparent_power", units="kW") == pytest.approx(
        [32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_rms_current():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesCurrentRMS(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "torque_out",
        np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCurrentRMS(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        [56.1, 62.3, 68.4, 73.9, 79.4, 84.2, 89.0, 93.1, 97.9, 102.0],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_rms_current_1_phase():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesCurrentRMS1Phase(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "ac_current_rms_in",
        np.array([56.1, 62.3, 68.4, 73.9, 79.4, 84.2, 89.0, 93.1, 97.9, 102.0]),
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesCurrentRMS1Phase(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_current_rms_in_one_phase", units="A") == pytest.approx(
        [18.7, 20.8, 22.8, 24.6, 26.5, 28.1, 29.7, 31.0, 32.6, 34.0],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_rms_voltage():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesVoltageRMS(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "apparent_power",
        np.array([32.5, 37.1, 41.7, 46.3, 50.9, 55.5, 60.2, 64.7, 69.4, 74.0]),
        units="kW",
    )
    ivc.add_output(
        "ac_current_rms_in",
        np.array([106.5, 118.2, 129.9, 140.3, 150.6, 159.7, 168.8, 176.6, 185.7, 193.5]),
        units="A",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltageRMS(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_rms_in", units="V") == pytest.approx(
        [305.1, 313.8, 321.0, 330.0, 337.9, 347.5, 356.6, 366.3, 373.7, 382.4],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_peak_voltage():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesVoltagePeak(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "ac_voltage_rms_in",
        np.array([101.7, 104.6, 107.0, 110.0, 112.7, 115.8, 118.9, 122.1, 124.6, 127.5]),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltagePeak(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_peak_in", units="V") == pytest.approx(
        [124.6, 128.1, 131.0, 134.7, 138.0, 141.8, 145.6, 149.5, 152.6, 156.2], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_maximum():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "torque_out",
        np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
        units="N*m",
    )
    ivc.add_output("rpm", np.linspace(3500, 4500, NB_POINTS_TEST), units="min**-1")
    ivc.add_output(
        "ac_voltage_peak_in",
        units="V",
        val=np.array([124.6, 128.1, 131.0, 134.7, 138.0, 141.8, 145.6, 149.5, 152.6, 156.2]),
    )
    ivc.add_output(
        "ac_current_rms_in_one_phase",
        np.array([56.1, 62.3, 68.4, 73.9, 79.4, 84.2, 89.0, 93.1, 97.9, 102.0]),
        units="A",
    )
    ivc.add_output(
        "power_losses",
        np.array([2.52, 2.66, 2.82, 2.98, 3.14, 3.3, 3.47, 3.63, 3.81, 3.98]),
        units="kW",
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, 10), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:current_ac_max", units="A"
    ) == pytest.approx(102.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:voltage_ac_max", units="V"
    ) == pytest.approx(156.2, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:torque_max", units="N*m"
    ) == pytest.approx(149.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:rpm_max", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:losses_max", units="kW"
    ) == pytest.approx(3.98, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:shaft_power_max", units="kW"
    ) == pytest.approx(70.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_pmsm():
    ivc = get_indep_var_comp(list_inputs(SizingPMSM(motor_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPMSM(motor_id="motor_1"), ivc)

    # om.n2(problem)

    assert problem.get_val("data:propulsion:he_power_train:PMSM:motor_1:diameter") == pytest.approx(
        0.268, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:PMSM:motor_1:length") == pytest.approx(
        0.06825, rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:torque_constant"
    ) == pytest.approx(2.09, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:loss_coefficient:alpha", units="W/N**2/m**2"
    ) == pytest.approx(0.0213, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:loss_coefficient:beta", units="W*s/rad"
    ) == pytest.approx(11.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:loss_coefficient:gamma", units="W*s**2/rad**2"
    ) == pytest.approx(0.0237, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:low_speed:CD0",
    ) * 1e3 == pytest.approx(0.357, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:CG:x", units="m"
    ) == pytest.approx(2.39, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:CG:y", units="m"
    ) == pytest.approx(1.5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performance_pmsm():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesPMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, 10), units="kW")
    ivc.add_output("rpm", np.linspace(3500, 4500, 10), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesPMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc)

    # om.n2(problem)

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        [56.1, 62.4, 68.3, 73.9, 79.2, 84.2, 89.0, 93.4, 97.7, 101.7], rel=1e-2
    )
    assert problem.get_val("ac_current_rms_in_one_phase", units="A") == pytest.approx(
        [18.7, 20.8, 22.8, 24.6, 26.4, 28.1, 29.7, 31.1, 32.6, 33.9], rel=1e-2
    )
    assert problem.get_val("ac_voltage_rms_in", units="V") == pytest.approx(
        [580.0, 594.8, 610.4, 626.4, 642.7, 659.3, 676.0, 692.9, 710.0, 727.1], rel=1e-2
    )
    assert problem.get_val("ac_voltage_peak_in", units="V") == pytest.approx(
        [710.4, 728.5, 747.6, 767.2, 787.1, 807.5, 827.9, 848.6, 869.6, 890.5], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_weight_per_fu():
    inputs_list = [
        "data:propulsion:he_power_train:PMSM:motor_1:mass",
        "data:environmental_impact:aircraft_per_fu",
        "data:TLAR:aircraft_lifespan",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PreLCAMotorProdWeightPerFU(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:mass_per_fu", units="kg"
    ) == pytest.approx(3.238e-5, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PMSM:motor_1:shaft_power_max",
        70.0,
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCPMSMCost(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:cost_per_unit", units="USD"
    ) == pytest.approx(6387.87, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_operational_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PMSM:motor_1:cost_per_unit",
        20815.61,
        units="USD",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LCCPMSMOperationalCost(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:operational_cost", units="USD/yr"
    ) == pytest.approx(1387.7, rel=1e-2)

    problem.check_partials(compact_print=True)
