# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

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
from ..components.sizing_power_density import SizingPowerDensity
from ..components.sizing_simple_pmsm_cg_x import SizingSimplePMSMCGX
from ..components.sizing_simple_pmsm_cg_y import SizingSimplePMSMCGY
from ..components.sizing_simple_pmsm_drag import SizingSimplePMSMDrag
from ..components.perf_torque import PerformancesTorque
from ..components.perf_losses import PerformancesLosses
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_current_rms import PerformancesCurrentRMS
from ..components.perf_current_rms_phase import PerformancesCurrentRMS1Phase
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum

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
from ..components.cstr_simple_pmsm import ConstraintPMSMPowerRateMission

from ..components.sizing_simple_pmsm import SizingSimplePMSM
from ..components.perf_simple_pmsm import PerformancesSimplePMSM

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
        "data:propulsion:he_power_train:simple_PMSM:motor_1:scaling:diameter"
    ) == pytest.approx(0.9, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_diameter():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorDiameter(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorDiameter(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:diameter", units="m"
    ) == pytest.approx(0.241, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorLengthScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLengthScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:scaling:length"
    ) == pytest.approx(0.97, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_length():
    ivc = get_indep_var_comp(list_inputs(SizingMotorLength(motor_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLength(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:length", units="m"
    ) == pytest.approx(0.088, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_weight():
    ivc = get_indep_var_comp(list_inputs(SizingMotorWeight(motor_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorWeight(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(16.19, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_power_density():
    ivc = get_indep_var_comp(
        list_inputs(SizingPowerDensity(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:mass",
        val=20.0,
        units="kg",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:shaft_power_rating",
        val=70.0,
        units="kW",
    )
    problem = run_system(SizingPowerDensity(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:power_density", units="kW/kg"
    ) == pytest.approx(3.5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorPhaseResistanceScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorPhaseResistanceScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:scaling:phase_resistance"
    ) == pytest.approx(0.93, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_resistance():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorPhaseResistance(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorPhaseResistance(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:phase_resistance", units="ohm"
    ) == pytest.approx(21.36e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque_constant_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorTorqueConstantScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorTorqueConstantScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:scaling:torque_constant"
    ) == pytest.approx(0.77, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_torque_constant():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorTorqueConstant(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorTorqueConstant(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:torque_constant", units="N*m/A"
    ) == pytest.approx(1.46, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_loss_coefficient_scaling():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorLossCoefficientScaling(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLossCoefficientScaling(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:scaling:alpha"
    ) == pytest.approx(1.57, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:scaling:beta"
    ) == pytest.approx(0.51, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:scaling:gamma"
    ) == pytest.approx(0.55, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_loss_coefficient():
    ivc = get_indep_var_comp(
        list_inputs(SizingMotorLossCoefficient(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingMotorLossCoefficient(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:loss_coefficient:alpha",
        units="W/N**2/m**2",
    ) == pytest.approx(0.025, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:loss_coefficient:beta", units="W*s/rad"
    ) == pytest.approx(3.38, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:loss_coefficient:gamma",
        units="W*s**2/rad**2",
    ) == pytest.approx(0.00825, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_motor_cg_x():
    expected_cg = [2.39, 0.25]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingSimplePMSMCGX(motor_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingSimplePMSMCGX(motor_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:simple_PMSM:motor_1:CG:x", units="m"
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_motor_cg_y():
    expected_cg = [1.5, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        ivc = get_indep_var_comp(
            list_inputs(SizingSimplePMSMCGY(motor_id="motor_1", position=option)),
            __file__,
            XML_FILE,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(SizingSimplePMSMCGY(motor_id="motor_1", position=option), ivc)

        assert problem.get_val(
            "data:propulsion:he_power_train:simple_PMSM:motor_1:CG:y", units="m"
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
                    SizingSimplePMSMDrag(
                        motor_id="motor_1", position=option, low_speed_aero=ls_option
                    )
                ),
                __file__,
                XML_FILE,
            )
            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingSimplePMSMDrag(motor_id="motor_1", position=option, low_speed_aero=ls_option),
                ivc,
            )

            if ls_option:
                assert problem.get_val(
                    "data:propulsion:he_power_train:simple_PMSM:motor_1:low_speed:CD0",
                ) * 1e3 == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:simple_PMSM:motor_1:cruise:CD0",
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
        "data:propulsion:he_power_train:simple_PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(150.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnforce(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_enforce():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnforce(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnforce(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:voltage_caliber", units="V"
    ) == pytest.approx(156.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_torque_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:simple_PMSM:motor_1:torque_rating", units="N*m"
    ) == pytest.approx(0.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_rpm_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsRPMEnsure(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsRPMEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:simple_PMSM:motor_1:rpm_rating", units="min**-1"
    ) == pytest.approx(-500.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_voltage_ensure():
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsVoltageEnsure(motor_id="motor_1")), __file__, XML_FILE
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsVoltageEnsure(motor_id="motor_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:simple_PMSM:motor_1:voltage_caliber", units="V"
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
        "data:propulsion:he_power_train:simple_PMSM:motor_1:shaft_power_rating", units="kW"
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
        list_inputs(PerformancesLosses(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "active_power",
        np.array([82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0]),
        units="W",
    )
    ivc.add_output(
        "shaft_power_out",
        np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]),
        units="W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesLosses(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("power_losses", units="W") == pytest.approx(
        [2.0, 1.0, 5.0, 8.0, 11.0, 13.0, 15.0, 16.0, 18.0, 19.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_active_power():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesActivePower(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 120, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesActivePower(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("active_power", units="kW") == pytest.approx(
        [31.9, 42.55, 53.19, 63.83, 74.47, 85.1, 95.74, 106.38, 117.02, 127.66], rel=1e-2
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
        "data:propulsion:he_power_train:simple_PMSM:motor_1:current_ac_max", units="A"
    ) == pytest.approx(102.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:voltage_ac_max", units="V"
    ) == pytest.approx(156.2, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:torque_max", units="N*m"
    ) == pytest.approx(149.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:rpm_max", units="min**-1"
    ) == pytest.approx(4500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:losses_max", units="kW"
    ) == pytest.approx(3.98, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:shaft_power_max", units="kW"
    ) == pytest.approx(70.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_pmsm():
    ivc = get_indep_var_comp(list_inputs(SizingSimplePMSM(motor_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingSimplePMSM(motor_id="motor_1"), ivc)

    # om.n2(problem)

    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:diameter"
    ) == pytest.approx(0.268, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:length"
    ) == pytest.approx(0.06825, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:torque_constant"
    ) == pytest.approx(2.09, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:loss_coefficient:alpha",
        units="W/N**2/m**2",
    ) == pytest.approx(0.0213, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:loss_coefficient:beta", units="W*s/rad"
    ) == pytest.approx(11.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:loss_coefficient:gamma",
        units="W*s**2/rad**2",
    ) == pytest.approx(0.0237, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:low_speed:CD0",
    ) * 1e3 == pytest.approx(0.357, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:CG:x", units="m"
    ) == pytest.approx(2.39, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:CG:y", units="m"
    ) == pytest.approx(1.5, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:simple_PMSM:motor_1:power_density", units="kW/kg"
    ) == pytest.approx(4.32, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performance_pmsm():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesSimplePMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, 10), units="kW")
    ivc.add_output("rpm", np.linspace(3500, 4500, 10), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSimplePMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc
    )

    # om.n2(problem)

    assert problem.get_val("ac_current_rms_in", units="A") == pytest.approx(
        [56.1, 62.4, 68.3, 73.9, 79.2, 84.2, 89.0, 93.4, 97.7, 101.7], rel=1e-2
    )
    assert problem.get_val("ac_current_rms_in_one_phase", units="A") == pytest.approx(
        [18.7, 20.8, 22.8, 24.6, 26.4, 28.1, 29.7, 31.1, 32.6, 33.9], rel=1e-2
    )
    assert problem.get_val("ac_voltage_rms_in", units="V") == pytest.approx(
        [
            569.27441347,
            587.34661707,
            605.41882067,
            623.49102428,
            641.56322788,
            659.63543148,
            677.70763508,
            695.77983869,
            713.85204229,
            731.92424589,
        ],
        rel=1e-2,
    )
    assert problem.get_val("ac_voltage_peak_in", units="V") == pytest.approx(
        [
            697.21591831,
            719.34975699,
            741.48359566,
            763.61743434,
            785.75127302,
            807.88511169,
            830.01895037,
            852.15278905,
            874.28662772,
            896.4204664,
        ],
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)
