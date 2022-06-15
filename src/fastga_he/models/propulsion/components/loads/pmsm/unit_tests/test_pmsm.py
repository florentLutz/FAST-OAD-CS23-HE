# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
import numpy as np
import pytest

from ..components.scaling import MotorScaling
from ..components.geometry import MotorGeometry
from ..components.weight import MotorWeight
from ..components.loss_coefficient_scaling import MotorLossCoefficientScaling
from ..components.loss_coefficient import MotorLossCoefficient
from ..components.optimal_torque import OptimalTorque

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_motor.xml"


def test_compute_scaling():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(MotorScaling(motor_id="motor_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(MotorScaling(motor_id="motor_1"), ivc)
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:scaling:diameter"] == pytest.approx(
        1.10, rel=1e-2
    )
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:scaling:length"] == pytest.approx(
        1.26, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compute_geometry():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(MotorGeometry(motor_id="motor_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(MotorGeometry(motor_id="motor_1"), ivc)
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:diameter"] == pytest.approx(
        0.429, rel=1e-2
    )
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:length"] == pytest.approx(
        0.284, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compute_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(MotorWeight(motor_id="motor_1")), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(MotorWeight(motor_id="motor_1"), ivc)
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:mass"] == pytest.approx(
        130.0, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_loss_coefficient_scaling():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(MotorLossCoefficientScaling(motor_id="motor_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(MotorLossCoefficientScaling(motor_id="motor_1"), ivc)
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:scaling:alpha"] == pytest.approx(
        0.54, rel=1e-2
    )
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:scaling:beta"] == pytest.approx(
        1.52, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_loss_coefficient():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(MotorLossCoefficient(motor_id="motor_1")), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(MotorLossCoefficient(motor_id="motor_1"), ivc)
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:alpha"] == pytest.approx(
        0.02268, rel=1e-2
    )
    assert problem["data:propulsion:he_power_train:PMSM:motor_1:beta"] == pytest.approx(
        0.7296, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_optimal_torque():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(OptimalTorque(motor_id="motor_1", number_of_points=10)), __file__, XML_FILE
    )
    ivc.add_output("power", np.linspace(50, 250, 10), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(OptimalTorque(motor_id="motor_1", number_of_points=10), ivc)
    assert problem["torque"] == pytest.approx(
        [256.0, 300.0, 336.0, 368.0, 397.0, 423.0, 447.0, 469.0, 490.0, 510.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)
