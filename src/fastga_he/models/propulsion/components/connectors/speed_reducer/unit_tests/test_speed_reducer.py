# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.perf_rpm_in import PerformancesRPMIn
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_torque_in import PerformancesTorqueIn
from ..components.perf_torque_out import PerformancesTorqueOut
from ..components.perf_maximum import PerformancesMaximum

from ..components.cstr_enforce import ConstraintsTorqueEnforce
from ..components.cstr_ensure import ConstraintsTorqueEnsure

from ..components.sizing_weight import SizingSpeedReducerWeight
from ..components.sizing_dimension_scaling import SizingSpeedReducerDimensionScaling
from ..components.sizing_dimension import SizingSpeedReducerDimensions
from ..components.sizing_cg_x import SizingSpeedReducerCGX
from ..components.sizing_cg_y import SizingSpeedReducerCGY

from ..components.perf_speed_reducer import PerformancesSpeedReducer
from ..components.sizing_speed_reducer import SizingSpeedReducer

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_speed_reducer.xml"
NB_POINTS_TEST = 10


def test_rpm_in():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRPMIn(speed_reducer_id="speed_reducer_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm_out", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRPMIn(speed_reducer_id="speed_reducer_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("rpm_in", units="min**-1") == pytest.approx(
        np.linspace(4000.0, 5000.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_shaft_power_in():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesShaftPowerIn(
                speed_reducer_id="speed_reducer_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("shaft_power_out", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesShaftPowerIn(
            speed_reducer_id="speed_reducer_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.linspace(103.1, 206.2, NB_POINTS_TEST), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_torque_in():

    ivc = om.IndepVarComp()
    ivc.add_output("shaft_power_in", val=np.linspace(103.1, 206.2, NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm_in", val=np.linspace(4000.0, 5000.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorqueIn(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.array([246.1, 266.0, 284.9, 302.9, 319.9, 336.1, 351.6, 366.3, 380.3, 393.8]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_torque_out():

    ivc = om.IndepVarComp()
    ivc.add_output("shaft_power_out", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")
    ivc.add_output("rpm_out", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesTorqueOut(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("torque_out", units="N*m") == pytest.approx(
        np.array([477.4, 516.1, 552.8, 587.6, 620.7, 652.1, 682.0, 710.6, 737.9, 763.9]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "torque_out",
        val=np.array([477.4, 516.1, 552.8, 587.6, 620.7, 652.1, 682.0, 710.6, 737.9, 763.9]),
        units="N*m",
    )
    ivc.add_output(
        "torque_in",
        val=np.array([246.1, 266.0, 284.9, 302.9, 319.9, 336.1, 351.6, 366.3, 380.3, 393.8]),
        units="N*m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(number_of_points=NB_POINTS_TEST, speed_reducer_id="speed_reducer_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:torque_in_max", units="N*m"
    ) == pytest.approx(393.8, rel=1e-3)
    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:torque_out_max", units="N*m"
    ) == pytest.approx(763.9, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_performances_speed_reducer():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesSpeedReducer(
                speed_reducer_id="speed_reducer_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm_out", val=np.linspace(2000.0, 2500.0, NB_POINTS_TEST), units="min**-1")
    ivc.add_output("shaft_power_out", val=np.linspace(100.0, 200.0, NB_POINTS_TEST), units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesSpeedReducer(
            speed_reducer_id="speed_reducer_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:torque_out_max", units="N*m"
    ) == pytest.approx(763.9, rel=1e-3)
    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.array([246.1, 266.0, 284.9, 302.9, 319.9, 336.1, 351.6, 366.3, 380.3, 393.8]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_torque_constraint_enforce():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnforce(speed_reducer_id="speed_reducer_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnforce(speed_reducer_id="speed_reducer_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:torque_in_rating", units="N*m"
    ) == pytest.approx(393.8, rel=1e-3)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:torque_out_rating",
            units="N*m",
        )
        == pytest.approx(763.9, rel=1e-3)
    )

    problem.check_partials(compact_print=True)


def test_torque_constraint_ensure():

    ivc = get_indep_var_comp(
        list_inputs(ConstraintsTorqueEnsure(speed_reducer_id="speed_reducer_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsTorqueEnsure(speed_reducer_id="speed_reducer_1"), ivc)

    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:speed_reducer:speed_reducer_1:torque_in_rating",
            units="N*m",
        )
        == pytest.approx(-6.2, rel=1e-3)
    )
    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:speed_reducer:speed_reducer_1:torque_out_rating",
            units="N*m",
        )
        == pytest.approx(-36.1, rel=1e-3)
    )

    problem.check_partials(compact_print=True)


def test_sizing_weight():

    ivc = get_indep_var_comp(
        list_inputs(SizingSpeedReducerWeight(speed_reducer_id="speed_reducer_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingSpeedReducerWeight(speed_reducer_id="speed_reducer_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:mass", units="kg"
    ) == pytest.approx(3.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_dimension_scaling():

    ivc = get_indep_var_comp(
        list_inputs(SizingSpeedReducerDimensionScaling(speed_reducer_id="speed_reducer_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingSpeedReducerDimensionScaling(speed_reducer_id="speed_reducer_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:scaling:dimensions"
    ) == pytest.approx(1.37, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_dimension():

    ivc = get_indep_var_comp(
        list_inputs(SizingSpeedReducerDimensions(speed_reducer_id="speed_reducer_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingSpeedReducerDimensions(speed_reducer_id="speed_reducer_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:width", units="m"
    ) == pytest.approx(0.205, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:height", units="m"
    ) == pytest.approx(0.205, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:length", units="m"
    ) == pytest.approx(0.363, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_cg_x():

    expected_cg = [2.69, 0.4, 4.80]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingSpeedReducerCGX(speed_reducer_id="speed_reducer_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingSpeedReducerCGX(speed_reducer_id="speed_reducer_1", position=option), ivc
        )

        assert (
            problem.get_val(
                "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:CG:x",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True)


def test_cg_y():

    expected_cg = [3.25, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_cg):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(SizingSpeedReducerCGY(speed_reducer_id="speed_reducer_1", position=option)),
            __file__,
            XML_FILE,
        )

        problem = run_system(
            SizingSpeedReducerCGY(speed_reducer_id="speed_reducer_1", position=option), ivc
        )

        assert (
            problem.get_val(
                "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:CG:y",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True)


def test_dc_dc_converter_sizing():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingSpeedReducer(speed_reducer_id="speed_reducer_1")),
        __file__,
        XML_FILE,
    )

    problem = run_system(SizingSpeedReducer(speed_reducer_id="speed_reducer_1"), ivc)

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:mass",
            units="kg",
        )
        == pytest.approx(3.11, rel=1e-2)
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:CG:x", units="m"
    ) == pytest.approx(2.69, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:CG:y", units="m"
    ) == pytest.approx(3.25, rel=1e-2)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:low_speed:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:speed_reducer:speed_reducer_1:cruise:CD0",
        )
        == pytest.approx(0.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)
