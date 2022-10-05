# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

from ..components.perf_torque import PerformancesTorque
from ..components.sizing_diameter_scaling import SizingMotorDiameterScaling
from ..components.sizing_diameter import SizingMotorDiameter
from ..components.sizing_length_scaling import SizingMotorLengthScaling
from ..components.sizing_length import SizingMotorLength
from ..components.sizing_weight import SizingMotorWeight
from ..components.sizing_resistance_scaling import SizingMotorPhaseResistanceScaling
from ..components.sizing_resistance import SizingMotorPhaseResistance

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_motor.xml"


def test_torque():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(PerformancesTorque(number_of_points=10)), __file__, XML_FILE
    )
    ivc.add_output("power", np.linspace(50, 250, 10), units="kW")
    ivc.add_output("rpm", np.linspace(2000, 3000, 10), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorque(number_of_points=10), ivc)

    assert problem.get_val("torque", units="N*m") == pytest.approx(
        [25.0, 34.0, 42.5, 50.0, 56.8, 63.0, 68.8, 74.0, 78.9, 83.0], rel=1e-2
    )

    problem.check_partials(compact_print=True)


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
    ) == pytest.approx(0.93308789, rel=1e-2)

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
