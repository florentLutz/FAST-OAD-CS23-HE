# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import openmdao.api as om
import pytest
import numpy as np

from ..components.sizing_tank_unusable_fuel import SizingFuelTankUnusableFuel
from ..components.sizing_tank_total_fuel_mission import SizingFuelTankTotalFuelMission
from ..components.sizing_tank_volume import SizingFuelTankVolume

from ..components.cstr_enforce import ConstraintsFuelTankCapacityEnforce
from ..components.cstr_ensure import ConstraintsFuelTankCapacityEnsure

from ..components.perf_fuel_mission_consumed import PerformancesFuelConsumedMission
from ..components.perf_fuel_remaining import PerformancesFuelRemainingMission

from ..components.perf_fuel_tanks import PerformancesFuelTank

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_tank.xml"
NB_POINTS_TEST = 10


def test_unusable_fuel_mission():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingFuelTankUnusableFuel(fuel_tank_id="fuel_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingFuelTankUnusableFuel(fuel_tank_id="fuel_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:unusable_fuel_mission", units="kg"
    ) == pytest.approx(1.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_total_fuel_mission():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingFuelTankTotalFuelMission(fuel_tank_id="fuel_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingFuelTankTotalFuelMission(fuel_tank_id="fuel_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_total_mission", units="kg"
    ) == pytest.approx(141.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_volume_fuel_tank():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingFuelTankVolume(fuel_tank_id="fuel_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingFuelTankVolume(fuel_tank_id="fuel_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:volume", units="L"
    ) == pytest.approx(393.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_enforce_tank_capacity():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFuelTankCapacityEnforce(fuel_tank_id="fuel_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsFuelTankCapacityEnforce(fuel_tank_id="fuel_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:capacity", units="kg"
    ) == pytest.approx(141.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_tank_capacity():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ConstraintsFuelTankCapacityEnsure(fuel_tank_id="fuel_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsFuelTankCapacityEnsure(fuel_tank_id="fuel_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "constraints:propulsion:he_power_train:fuel_tank:fuel_tank_1:capacity", units="kg"
    ) == pytest.approx(-141.4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_consumed_mission():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesFuelConsumedMission(
            fuel_tank_id="fuel_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission", units="kg"
    ) == pytest.approx(276.85, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuel_remaining_mission():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesFuelRemainingMission(
                fuel_tank_id="fuel_tank_1", number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("fuel_consumed_t", val=np.full(NB_POINTS_TEST, 14.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesFuelRemainingMission(
            fuel_tank_id="fuel_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("fuel_remaining_t", units="kg") == pytest.approx(
        np.array([140.0, 126.0, 112.0, 98.0, 84.0, 70.0, 56.0, 42.0, 28.0, 14.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performances_fuel_tank():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesFuelTank(fuel_tank_id="fuel_tank_1", number_of_points=NB_POINTS_TEST),
        ivc,
    )
    assert problem.get_val("fuel_remaining_t", units="kg") == pytest.approx(
        np.array([276.85, 263.48, 246.93, 227.2, 204.28, 178.19, 148.91, 116.46, 80.82, 42.0]),
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission", units="kg"
    ) == pytest.approx(279.62, rel=1e-2)
