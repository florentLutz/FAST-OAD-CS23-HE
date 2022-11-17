# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

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
from ..components.perf_torque import PerformancesTorque
from ..components.perf_losses import PerformancesLosses
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_current_rms import PerformancesCurrentRMS
from ..components.perf_current_rms_phase import PerformancesCurrentRMS1Phase
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_voltage_peak import PerformancesVoltagePeak

from ..components.sizing_pmsm import SizingPMSM
from ..components.perf_pmsm import PerformancePMSM

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
    ) == pytest.approx(21.36, rel=1e-2)

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


def test_torque():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(PerformancesTorque(number_of_points=NB_POINTS_TEST)), __file__, XML_FILE
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm", np.linspace(3500, 4500, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque", units="N*m") == pytest.approx(
        [82.0, 91.0, 100.0, 108.0, 116.0, 123.0, 130.0, 136.0, 143.0, 149.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_losses():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesLosses(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "torque",
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
        list_inputs(PerformancesEfficiency(number_of_points=NB_POINTS_TEST)), __file__, XML_FILE
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, NB_POINTS_TEST), units="kW")
    ivc.add_output(
        "power_losses",
        np.array([2.52, 2.66, 2.82, 2.98, 3.14, 3.3, 3.47, 3.63, 3.81, 3.98]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesEfficiency(number_of_points=NB_POINTS_TEST), ivc)

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
        "torque",
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


def test_sizing_pmsm():

    ivc = get_indep_var_comp(list_inputs(SizingPMSM(motor_id="motor_1")), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPMSM(motor_id="motor_1"), ivc)

    # om.n2(problem)

    assert problem.get_val("data:propulsion:he_power_train:PMSM:motor_1:diameter") == pytest.approx(
        0.24, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:PMSM:motor_1:length") == pytest.approx(
        0.089, rel=1e-2
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:torque_constant"
    ) == pytest.approx(1.45, rel=1e-2)
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


def test_performance_pmsm():

    ivc = get_indep_var_comp(
        list_inputs(PerformancePMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", np.linspace(30, 70, 10), units="kW")
    ivc.add_output("rpm", np.linspace(3500, 4500, 10), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancePMSM(motor_id="motor_1", number_of_points=NB_POINTS_TEST), ivc)

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
