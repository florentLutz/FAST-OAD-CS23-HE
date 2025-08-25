"""
Test mission vector module.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import os.path as pth
import pytest
import copy
import warnings

import numpy as np
import openmdao.api as om

import plotly.graph_objects as go

import fastoad.api as oad

import fastga_he.api as oad_he

from fastga_he.models.performances.mission_vector.initialization.initialize_altitude import (
    InitializeAltitude,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_density import (
    InitializeDensity,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_temperature import (
    InitializeTemperature,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_climb_airspeed import (
    InitializeClimbAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_descent_airspeed import (
    InitializeDescentAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_reserve_airspeed import (
    InitializeReserveAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_airspeed import (
    InitializeAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_vertical_airspeed import (
    InitializeVerticalAirspeed,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_gamma import (
    InitializeGamma,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_airspeed_derivatives import (
    InitializeAirspeedDerivatives,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_time_and_distance import (
    InitializeTimeAndDistance,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_time_step import (
    InitializeTimeStep,
)

from fastga_he.models.performances.mission_vector.mission.performance_per_phase import (
    PerformancePerPhase,
)
from fastga_he.models.performances.mission_vector.mission.sizing_time import SizingDuration
from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)

from fastga_he.models.performances.mission_vector.initialization.initialize_cg import InitializeCoG
from fastga_he.models.performances.mission_vector.mission_vector import MissionVector
from fastga_he.models.propulsion.assemblers.sizing_from_pt_file import PowerTrainSizingFromFile
from fastga_he.models.performances.op_mission_vector.update_tow import UpdateTOW

from fastga_he.models.performances.payload_range.payload_range import ComputePayloadRange
from fastga_he.models.performances.payload_range.mission_range_from_soc import (
    OperationalMissionVectorWithTargetSoC,
)

from fastga_he.models.performances.payload_range.payload_range_inner_sampling import (
    ComputePayloadRangeInnerSampling,
)
from fastga_he.models.performances.payload_range.payload_range_inner import ComputePayloadRangeInner
from fastga_he.models.performances.payload_range.payload_range_inner_group import (
    ComputePayloadRangeInnerGroup,
)

from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
XML_FILE = "sample_ac.xml"
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(autouse=False)
def catch_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        yield

        for warning in w:
            print(warning)


@pytest.fixture()
def restore_submodels():
    """
    Since the submodels in the configuration file differ from the defaults, this restore process
    ensures subsequent assembly tests run under default conditions.
    """
    old_submodels = copy.deepcopy(oad.RegisterSubmodel.active_models)
    yield
    oad.RegisterSubmodel.active_models = old_submodels


def test_initialize_altitude():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeAltitude(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        InitializeAltitude(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_altitude = np.array(
        [
            0.0,
            270.9,
            541.9,
            812.8,
            1083.7,
            1354.7,
            1625.6,
            1896.5,
            2167.5,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            1828.8,
            1219.2,
            609.6,
            0.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        ]
    )
    assert np.max(np.abs(problem.get_val("altitude", units="m") - expected_altitude)) <= 1e-1

    problem.check_partials(compact_print=True)


def test_initialize_density():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    altitude = np.array(
        [
            0.0,
            270.9,
            541.9,
            812.8,
            1083.7,
            1354.7,
            1625.6,
            1896.5,
            2167.5,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            1828.8,
            1219.2,
            609.6,
            0.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        ]
    )
    ivc.add_output("altitude", units="m", val=altitude)

    problem = run_system(
        InitializeDensity(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_density = np.array(
        [
            1.224,
            1.193,
            1.162,
            1.132,
            1.102,
            1.073,
            1.044,
            1.016,
            0.989,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
            1.023,
            1.087,
            1.154,
            1.224,
            1.111,
            1.111,
            1.111,
            1.111,
            1.111,
        ]
    )
    assert problem.get_val("density", units="kg/m**3") == pytest.approx(expected_density, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_initialize_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    altitude = np.array(
        [
            0.0,
            270.9,
            541.9,
            812.8,
            1083.7,
            1354.7,
            1625.6,
            1896.5,
            2167.5,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            2438.4,
            1828.8,
            1219.2,
            609.6,
            0.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        ]
    )
    ivc.add_output("altitude", units="m", val=altitude)

    problem = run_system(
        InitializeTemperature(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_temperature = np.array(
        [
            288.15,
            286.38,
            284.62,
            282.86,
            281.10,
            279.34,
            277.58,
            275.82,
            274.06,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            272.30,
            276.26,
            280.22,
            284.18,
            288.15,
            281.65,
            281.65,
            281.65,
            281.65,
            281.65,
        ]
    )
    assert problem.get_val("exterior_temperature", units="degK") == pytest.approx(
        expected_temperature, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_climb_speed():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeClimbAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("mass", units="kg", val=np.full(30, 1700.0))

    problem = run_system(
        InitializeClimbAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    climb_eas = problem.get_val("data:mission:sizing:main_route:climb:v_eas", units="m/s")
    assert climb_eas == pytest.approx(44.48, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_descent_speed():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeDescentAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("mass", units="kg", val=np.full(30, 1700.0))
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )

    problem = run_system(
        InitializeDescentAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    descent_eas = problem.get_val("data:mission:sizing:main_route:descent:v_eas", units="m/s")
    assert descent_eas == pytest.approx(56.27, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_reserve_speed():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeReserveAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("mass", units="kg", val=np.full(30, 1700.0))

    problem = run_system(
        InitializeReserveAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    reserve_tas = problem.get_val("data:mission:sizing:main_route:reserve:v_tas", units="m/s")
    assert reserve_tas == pytest.approx(46.69, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_initialize_airspeed():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )

    problem = run_system(
        InitializeAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )
    expected_tas = np.array(
        [
            45.1,
            45.7,
            46.3,
            46.9,
            47.5,
            48.2,
            48.8,
            49.5,
            50.2,
            50.9,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            82.3,
            64.3,
            62.4,
            60.5,
            58.8,
            57.1,
            47.3,
            47.3,
            47.3,
            47.3,
            47.3,
        ]
    )
    expected_eas = np.array(
        [
            45.1,
            45.1,
            45.1,
            45.1,
            45.1,
            45.1,
            45.1,
            45.1,
            45.1,
            45.1,
            73.0,
            73.0,
            73.0,
            73.0,
            73.0,
            73.0,
            73.0,
            73.0,
            73.0,
            73.0,
            57.1,
            57.1,
            57.1,
            57.1,
            57.1,
            45.1,
            45.1,
            45.1,
            45.1,
            45.1,
        ]
    )
    assert problem.get_val("true_airspeed", units="m/s") == pytest.approx(expected_tas, rel=1e-2)
    assert problem.get_val("equivalent_airspeed", units="m/s") == pytest.approx(
        expected_eas, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_initialize_airspeed_derivatives():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.array(
            [
                44.4,
                45.0,
                45.6,
                46.2,
                46.8,
                47.5,
                48.1,
                48.8,
                49.4,
                50.1,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.4,
                61.5,
                59.7,
                57.9,
                56.2,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )
    ivc.add_output(
        "equivalent_airspeed",
        units="m/s",
        val=np.array(
            [
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                44.48,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                72.97,
                56.27,
                56.27,
                56.27,
                56.27,
                56.27,
                57.15,
                57.15,
                57.15,
                57.15,
                57.15,
            ]
        ),
    )
    ivc.add_output(
        "gamma",
        units="deg",
        val=np.array(
            [
                2.0,
                1.9,
                1.8,
                1.7,
                1.6,
                1.5,
                1.4,
                1.3,
                1.2,
                1.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    problem = run_system(
        InitializeAirspeedDerivatives(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )
    expected_derivative = np.array(
        [
            0.00330842,
            0.0032472,
            0.00317812,
            0.00310071,
            0.00301457,
            0.00292544,
            0.00282017,
            0.0027103,
            0.00258387,
            0.00245106,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.00356722,
            -0.00330743,
            -0.00307075,
            -0.00285019,
            -0.00264925,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert problem.get_val("d_vx_dt", units="m/s**2") == pytest.approx(
        expected_derivative, abs=1e-1
    )

    problem.check_partials(compact_print=True)


def test_initialize_vertical_airspeed():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeVerticalAirspeed(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )

    problem = run_system(
        InitializeVerticalAirspeed(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_vertical_speed = np.array(
        [
            6.096,
            5.813,
            5.531,
            5.249,
            4.967,
            4.684,
            4.402,
            4.120,
            3.838,
            3.556,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.524,
            -1.524,
            -1.524,
            -1.524,
            -1.524,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert problem.get_val("vertical_speed", units="m/s") == pytest.approx(
        expected_vertical_speed, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_initialize_gamma():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "vertical_speed",
        units="m/s",
        val=np.array(
            [
                6.096,
                5.813,
                5.531,
                5.249,
                4.967,
                4.684,
                4.402,
                4.120,
                3.838,
                3.556,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.524,
                -1.524,
                -1.524,
                -1.524,
                -1.524,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.array(
            [
                44.4,
                45.0,
                45.6,
                46.2,
                46.8,
                47.5,
                48.1,
                48.8,
                49.4,
                50.1,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.4,
                61.5,
                59.7,
                57.9,
                56.2,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )

    problem = run_system(
        InitializeGamma(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_gamma = np.array(
        [
            0.1377,
            0.1295,
            0.1216,
            0.1138,
            0.1063,
            0.0987,
            0.0916,
            0.0845,
            0.0777,
            0.0710,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.024,
            -0.024,
            -0.025,
            -0.026,
            -0.027,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert problem.get_val("gamma", units="rad") == pytest.approx(expected_gamma, abs=1e-3)

    problem.check_partials(compact_print=True)


def test_initialize_time_and_position():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeTimeAndDistance(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "altitude",
        units="m",
        val=np.array(
            [
                0.0,
                270.9,
                541.9,
                812.8,
                1083.7,
                1354.7,
                1625.6,
                1896.5,
                2167.5,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                2438.4,
                1828.8,
                1219.2,
                609.6,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        ),
    )
    ivc.add_output(
        "true_airspeed",
        units="m/s",
        val=np.array(
            [
                44.4,
                45.0,
                45.6,
                46.2,
                46.8,
                47.5,
                48.1,
                48.8,
                49.4,
                50.1,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.4,
                61.5,
                59.7,
                57.9,
                56.2,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )
    ivc.add_output(
        "horizontal_speed",
        units="m/s",
        val=np.array(
            [
                43.9,
                44.6,
                45.2,
                45.9,
                46.5,
                47.2,
                47.8,
                48.6,
                49.2,
                49.9,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                82.3,
                63.3,
                61.4,
                59.6,
                57.8,
                56.1,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ]
        ),
    )

    problem = run_system(
        InitializeTimeAndDistance(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_position = np.array(
        [
            0.0,
            1.087,
            2.245,
            3.481,
            4.804,
            6.225,
            7.754,
            9.408,
            11.206,
            13.167,
            19.559,
            25.951,
            32.342,
            38.734,
            45.126,
            51.518,
            57.91,
            64.302,
            70.694,
            77.085,
            83.477,
            96.944,
            110.011,
            122.689,
            134.989,
            148.792,
            162.595,
            176.398,
            190.2,
            204.003,
        ]
    )
    assert problem.get_val("position", units="nmi") == pytest.approx(expected_position, rel=1e-3)
    expected_time = np.array(
        [
            0.0,
            0.758,
            1.554,
            2.392,
            3.276,
            4.212,
            5.205,
            6.265,
            7.4,
            8.621,
            11.018,
            13.415,
            15.812,
            18.209,
            20.606,
            23.003,
            25.4,
            27.797,
            30.194,
            32.591,
            34.988,
            41.654,
            48.321,
            54.988,
            61.654,
            70.654,
            79.654,
            88.654,
            97.654,
            106.654,
        ]
    )
    assert problem.get_val("time", units="min") == pytest.approx(expected_time, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_initialize_time_step():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "time",
        units="min",
        val=np.array(
            [
                0.0,
                0.758,
                1.554,
                2.392,
                3.276,
                4.212,
                5.205,
                6.265,
                7.4,
                8.621,
                11.018,
                13.415,
                15.812,
                18.209,
                20.606,
                23.003,
                25.4,
                27.797,
                30.194,
                32.591,
                34.988,
                41.654,
                48.321,
                54.988,
                61.654,
                70.654,
                79.654,
                88.654,
                97.654,
                106.654,
            ]
        ),
    )

    problem = run_system(
        InitializeTimeStep(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    expected_time_step = np.array(
        [
            0.758,
            0.796,
            0.838,
            0.884,
            0.936,
            0.993,
            1.06,
            1.135,
            1.221,
            1.221,
            2.397,
            2.397,
            2.397,
            2.397,
            2.397,
            2.397,
            2.397,
            2.397,
            2.397,
            2.397,
            6.666,
            6.667,
            6.667,
            6.666,
            6.666,
            9.0,
            9.0,
            9.0,
            9.0,
            9.0,
        ]
    )
    assert problem.get_val("time_step", units="min") == pytest.approx(expected_time_step, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_initialize_cog():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeCoG(
                number_of_points_climb=10,
                number_of_points_cruise=10,
                number_of_points_descent=5,
                number_of_points_reserve=5,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("fuel_mass_t", units="kg", val=np.linspace(45.0, 2.0, 30))
    ivc.add_output("fuel_lever_arm_t", units="kg*m", val=np.linspace(45.0, 2.0, 30) * 2.5)

    problem = run_system(
        InitializeCoG(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )
    expected_cog = np.array(
        [
            3.45,
            3.451,
            3.451,
            3.452,
            3.453,
            3.454,
            3.455,
            3.456,
            3.457,
            3.458,
            3.459,
            3.46,
            3.461,
            3.462,
            3.463,
            3.464,
            3.464,
            3.465,
            3.466,
            3.467,
            3.468,
            3.469,
            3.47,
            3.471,
            3.472,
            3.473,
            3.474,
            3.475,
            3.476,
            3.477,
        ]
    )
    assert problem.get_val("x_cg", units="m") == pytest.approx(expected_cog, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_performances_per_phase():
    ivc = om.IndepVarComp()

    ivc.add_output(
        name="time",
        val=np.array(
            [
                0.0,
                0.758,
                1.554,
                2.392,
                3.276,
                4.212,
                5.205,
                6.265,
                7.4,
                8.621,
                11.018,
                13.415,
                15.812,
                18.209,
                20.606,
                23.003,
                25.4,
                27.797,
                30.194,
                32.591,
                34.988,
                41.654,
                48.321,
                54.988,
                61.654,
                70.654,
                79.654,
                88.654,
                97.654,
                106.654,
            ]
        ),
        units="min",
    )
    ivc.add_output(
        name="position",
        val=np.array(
            [
                0.0,
                1.087,
                2.245,
                3.481,
                4.804,
                6.225,
                7.754,
                9.408,
                11.206,
                13.167,
                19.559,
                25.951,
                32.342,
                38.734,
                45.126,
                51.518,
                57.91,
                64.302,
                70.694,
                77.085,
                83.477,
                96.944,
                110.011,
                122.689,
                134.989,
                148.792,
                162.595,
                176.398,
                190.2,
                204.003,
            ]
        ),
        units="nmi",
    )
    ivc.add_output(
        name="fuel_consumed_t_econ",
        val=np.linspace(1.0, 2.0, 32),
        units="kg",
    )
    ivc.add_output(
        name="fuel_mass_t_econ",
        val=np.linspace(48.0, 2.0, 32),
        units="kg",
    )
    ivc.add_output(
        name="fuel_lever_arm_t_econ",
        val=np.linspace(144.0, 6.0, 32),
        units="kg*m",
    )
    ivc.add_output(
        name="non_consumable_energy_t_econ",
        val=np.linspace(10.0, 25.0, 32),
        units="W*h",
    )
    ivc.add_output(
        name="thrust_rate_t_econ",
        val=np.ones(32),
    )

    problem = run_system(
        PerformancePerPhase(
            number_of_points_climb=10,
            number_of_points_cruise=10,
            number_of_points_descent=5,
            number_of_points_reserve=5,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:mission:sizing:main_route:climb:duration", units="min"
    ) == pytest.approx(11.018, rel=1e-3)
    assert problem.get_val(
        "data:mission:sizing:main_route:cruise:duration", units="min"
    ) == pytest.approx(23.97, rel=1e-3)
    assert problem.get_val(
        "data:mission:sizing:main_route:descent:duration", units="min"
    ) == pytest.approx(35.666, rel=1e-3)

    assert problem.get_val(
        "data:mission:sizing:main_route:climb:distance", units="nmi"
    ) == pytest.approx(19.559, rel=1e-3)
    assert problem.get_val(
        "data:mission:sizing:main_route:cruise:distance", units="nmi"
    ) == pytest.approx(63.918, rel=1e-3)
    assert problem.get_val(
        "data:mission:sizing:main_route:descent:distance", units="nmi"
    ) == pytest.approx(65.315, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_sizing_duration():
    inputs_list = [
        "data:mission:sizing:main_route:climb:duration",
        "data:mission:sizing:main_route:cruise:duration",
        "data:mission:sizing:main_route:descent:duration",
        "data:mission:sizing:main_route:reserve:duration",
        "data:mission:sizing:taxi_out:duration",
        "data:mission:sizing:taxi_in:duration",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    problem = run_system(SizingDuration(), ivc)

    sizing_duration = problem.get_val("data:mission:sizing:duration", units="h")
    assert sizing_duration == pytest.approx(2.01279002, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_mission_vector(restore_submodels):
    # Research independent input value in .xml file
    oad.RegisterSubmodel.active_models[HE_SUBMODEL_ENERGY_CONSUMPTION] = (
        "fastga_he.submodel.performances.energy_consumption.basic"
    )
    oad.RegisterSubmodel.active_models[HE_SUBMODEL_DEP_EFFECT] = (
        "fastga_he.submodel.performances.dep_effect.none"
    )

    ivc = get_indep_var_comp(
        list_inputs(
            MissionVector(
                number_of_points_climb=30,
                number_of_points_cruise=30,
                number_of_points_descent=20,
                number_of_points_reserve=10,
                use_linesearch=False,
            )
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        MissionVector(
            number_of_points_climb=30,
            number_of_points_cruise=30,
            number_of_points_descent=20,
            number_of_points_reserve=10,
            use_linesearch=False,
        ),
        ivc,
    )
    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(45.63, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "n2_simple_mission_vector.html"))

    problem.check_partials(compact_print=True)


def test_mission_vector_from_yml():
    # Define used files depending on options
    xml_file_name = "sample_ac.xml"
    process_file_name = "mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(157.22, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.01359, abs=1e-2)
    pt_mass = problem.get_val("data:propulsion:he_power_train:mass", units="kg")
    assert pt_mass == pytest.approx(1254.45, abs=1e-2)


def test_op_mission_vector_from_yml():
    # Define used files depending on options
    xml_file_name = "op_mission_inputs.xml"
    process_file_name = "op_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(
    #     problem,
    #     outfile=pth.join(RESULTS_FOLDER_PATH, "n2_op_mission_vector_from_yml.html"),
    #     show_browser=False,
    # )

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:operational:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:operational:energy", units="kW*h")
    assert sizing_energy == pytest.approx(99.22, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(37.47, abs=1e-2)
    mission_tow = problem.get_val("data:mission:operational:TOW", units="kg")
    assert mission_tow == pytest.approx(840.0, abs=1e-2)

    co2_emissions_ice = problem.get_val(
        "data:environmental_impact:operation:operational:he_power_train:battery_pack:battery_pack_1:CO2",
        units="kg",
    )
    assert co2_emissions_ice == pytest.approx(0.0, abs=1e-2)


def test_update_tow():
    ivc = get_indep_var_comp(list_inputs(UpdateTOW()), __file__, XML_FILE)

    problem = run_system(UpdateTOW(), ivc)
    sizing_fuel = problem.get_val("data:mission:operational:TOW", units="kg")
    assert sizing_fuel == pytest.approx(950.0, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_mission_vector_from_yml_gearbox():
    # Define used files depending on options
    xml_file_name = "sample_ac.xml"
    process_file_name = "mission_vector_gearbox.yml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_speed_reducer.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "simple_assembly_speed_reducer.html")

    power_train_network_viewer(pt_file_path, network_file_path)

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    # Efficiency of the gearbox is 0.98, if we take the result from the previous test and include
    # the efficiency of the gearbox assuming nothing else changes we should an energy consumed of
    # 146.75/0.98 = 149.74, which means we must also have impacted the efficiencies of other
    # components of the pt
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(148.60, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.01359, abs=1e-2)
    pt_mass = problem.get_val("data:propulsion:he_power_train:mass", units="kg")
    assert pt_mass == pytest.approx(1152.75, abs=1e-2)


def test_mission_vector_from_yml_simplified_models():
    previously_active_models = copy.deepcopy(oad.RegisterSubmodel.active_models)

    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.performances.dc_line.temperature_profile"
    ] = "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.constant"
    oad.RegisterSubmodel.active_models["submodel.propulsion.dc_dc_converter.efficiency"] = (
        "fastga_he.submodel.propulsion.dc_dc_converter.efficiency.fixed"
    )
    oad.RegisterSubmodel.active_models["submodel.propulsion.inverter.junction_temperature"] = (
        "fastga_he.submodel.propulsion.inverter.junction_temperature.fixed"
    )
    oad.RegisterSubmodel.active_models["submodel.propulsion.inverter.efficiency"] = (
        "fastga_he.submodel.propulsion.inverter.efficiency.fixed"
    )

    # Define used files depending on options
    xml_file_name = "sample_ac.xml"
    process_file_name = "mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(157.22, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.0082, abs=1e-2)

    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.performances.dc_line.temperature_profile"
    ] = previously_active_models["submodel.propulsion.performances.dc_line.temperature_profile"]
    oad.RegisterSubmodel.active_models["submodel.propulsion.dc_dc_converter.efficiency"] = (
        previously_active_models["submodel.propulsion.dc_dc_converter.efficiency"]
    )
    oad.RegisterSubmodel.active_models["submodel.propulsion.inverter.junction_temperature"] = (
        previously_active_models["submodel.propulsion.inverter.junction_temperature"]
    )
    oad.RegisterSubmodel.active_models["submodel.propulsion.inverter.efficiency"] = (
        previously_active_models["submodel.propulsion.inverter.efficiency"]
    )


def test_mission_vector_direct_bus_battery_connection():
    # Define used files depending on options
    xml_file_name = "sample_ac.xml"
    process_file_name = "mission_vector_direct_bus_battery.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(141.73, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.0439, abs=1e-2)
    pt_mass = problem.get_val("data:propulsion:he_power_train:mass", units="kg")
    assert pt_mass == pytest.approx(910.533, abs=1e-2)


def test_mission_vector_direct_sspc_battery_connection():
    # Define used files depending on options
    xml_file_name = "sample_ac.xml"
    process_file_name = "mission_vector_direct_sspc_battery.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(0.0, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(143.06, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.0349, abs=1e-2)
    pt_mass = problem.get_val("data:propulsion:he_power_train:mass", units="kg")
    assert pt_mass == pytest.approx(924.43, abs=1e-2)
    co2_emissions = problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:battery_pack:battery_pack_1:CO2",
        units="kg",
    )
    assert co2_emissions == pytest.approx(0.0, abs=1e-2)


def test_mission_vector_from_yml_fuel():
    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "fuel_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(33.42, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)


def test_mission_vector_from_yml_two_fuel():
    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion_criss_cross.xml"
    process_file_name = "two_fuel_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(31.67, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)


def test_mission_vector_from_yml_fuel_turbo():
    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "fuel_turbo_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(38.13, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)


def test_mission_vector_from_yml_fuel_and_battery():
    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "fuel_and_battery_propulsion_mission_vector.yml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "fuel_and_battery_propulsion.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "fuel_and_battery.html")

    power_train_network_viewer(pt_file_path, network_file_path)

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(19.72, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(73.288, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(0.1254, abs=1e-2)
    co2_emissions_ice = problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:ICE:ice_1:CO2", units="kg"
    )
    assert co2_emissions_ice == pytest.approx(61.16, abs=1e-2)


def test_mission_vector_from_yml_fuel_and_battery_gear():
    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "fuel_and_battery_propulsion_gear_mission_vector.yml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "fuel_and_battery_propulsion_gear.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "fuel_and_battery_propulsion_gear.html")

    power_train_network_viewer(pt_file_path, network_file_path)

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(27.56, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(30.76, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.48, abs=1e-2)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_recording():
    oad.RegisterSubmodel.active_models["submodel.propulsion.constraints.pmsm.rpm"] = (
        "fastga_he.submodel.propulsion.constraints.pmsm.rpm.ensure"
    )
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.battery.state_of_charge"
    ] = "fastga_he.submodel.propulsion.constraints.battery.state_of_charge.enforce"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.performances.dc_line.temperature_profile"
    ] = "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.with_dynamics"
    oad.RegisterSubmodel.active_models["submodel.propulsion.constraints.inverter.current"] = (
        "fastga_he.submodel.propulsion.constraints.inverter.current.enforce"
    )
    oad.RegisterSubmodel.active_models["submodel.propulsion.constraints.pmsm.torque"] = (
        "fastga_he.submodel.propulsion.constraints.pmsm.torque.enforce"
    )
    oad.RegisterSubmodel.active_models["submodel.propulsion.constraints.generator.rpm"] = (
        "fastga_he.submodel.propulsion.constraints.generator.rpm.ensure"
    )

    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"

    problem = om.Problem(reports=False)
    model = problem.model

    group = om.Group()
    group.add_subsystem(
        "pt_sizing",
        PowerTrainSizingFromFile(
            power_train_file_path="data/fuel_and_battery_propulsion.yml",
        ),
        promotes=["*"],
    )
    group.add_subsystem(
        "mission_vector",
        MissionVector(
            number_of_points_climb=30,
            number_of_points_cruise=30,
            number_of_points_descent=20,
            number_of_points_reserve=10,
            power_train_file_path="data/fuel_and_battery_propulsion.yml",
            out_file="results/mission_data_fuel_and_battery.csv",
            use_linesearch=False,
        ),
        promotes=["*"],
    )

    ivc = get_indep_var_comp(
        list_inputs(group),
        __file__,
        xml_file_name,
    )

    # Create a new problem
    model.add_subsystem("data", ivc, promotes=["*"])
    model.add_subsystem("group", group, promotes=["*"])
    model.nonlinear_solver = om.NonlinearBlockGS(
        maxiter=10, iprint=2, rtol=1e-5, debug_print=True, reraise_child_analysiserror=True
    )
    model.linear_solver = om.LinearBlockGS()

    problem.setup()

    # Adding a recorder
    recorder = om.SqliteRecorder("results/cases.sql")
    solver = model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True

    problem.run_model()
    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(19.72, abs=1e-2)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_case_reader():
    fig = go.Figure()
    cr = om.CaseReader("results/cases.sql")

    solver_case = cr.get_cases("root.nonlinear_solver")
    for i, case in enumerate(solver_case):
        battery_soc = case[
            "solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances.battery_pack_1.state_of_charge"
        ]

        scatter = go.Scatter(
            x=np.arange(len(battery_soc)),
            y=battery_soc,
            mode="lines+markers",
            name="Battery SOC during case " + str(i),
        )
        fig.add_trace(scatter)

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_case_analyzer():
    cr = om.CaseReader("results/cases.sql")

    solver_case = cr.get_cases("root.nonlinear_solver")
    for case in solver_case:
        residuals_dict = {}

        for residual in case.residuals:
            residuals_dict[residual] = sum(np.square(case.residuals[residual]))

        top_residuals = max(residuals_dict, key=residuals_dict.get)

        # print the result
        print(
            f"The top residuals is {top_residuals} with a score of {residuals_dict[top_residuals]}."
        )


def test_criss_cross_network_viewer():
    # Define used files depending on options
    pt_file_name = "fuel_and_battery_propulsion_criss_cross.yml"

    oad_he.power_train_network_viewer(
        power_train_file_path=pth.join(DATA_FOLDER_PATH, pt_file_name),
        network_file_path=pth.join(RESULTS_FOLDER_PATH, "criss_cross.html"),
    )


def test_mission_criss_cross():
    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion_criss_cross.xml"
    process_file_name = "fuel_and_battery_propulsion_criss_cross_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(24.69, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(48.180, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.0153, abs=1e-2)


def test_mission_vector_turboshaft():
    # Define used files depending on options
    xml_file_name = "sample_turboshaft_propulsion.xml"
    process_file_name = "turboshaft_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # om.n2(problem, show_browser=True)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(87.76, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)
    co2_emissions_ice = problem.get_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2",
        units="kg",
    )
    assert co2_emissions_ice == pytest.approx(276.91, abs=1e-2)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_mission_vector_eight_propeller_with_turned_off_sspc():
    # Define used files depending on options
    xml_file_name = "octo_assembly.xml"
    process_file_name = "octo_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    model = problem.model.performances.solve_equilibrium.compute_dep_equilibrium
    recorder = om.SqliteRecorder(pth.join(RESULTS_FOLDER_PATH, "cases_octo_prop.sql"))
    solver = model.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True
    solver.recording_options["record_outputs"] = True

    # om.n2(problem, show_browser=True)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(15.15, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_4:torque_max", units="N*m"
    ) == pytest.approx(0.0, abs=1e-2)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_case_analyzer_octo_prop():
    cr = om.CaseReader(pth.join(RESULTS_FOLDER_PATH, "cases_octo_prop.sql"))

    solver_case = cr.get_cases(
        "root.performances.solve_equilibrium.compute_dep_equilibrium.nonlinear_solver"
    )
    for case in solver_case:
        residuals_dict = {}

        for residual in case.residuals:
            residuals_dict[residual] = sum(np.square(case.residuals[residual]))

        top_residuals = max(residuals_dict, key=residuals_dict.get)

        # print the result
        print(
            f"The top residuals is {top_residuals} with a score of {residuals_dict[top_residuals]}."
        )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_residuals_viewer():
    """Inspection of the residuals."""

    cr = om.CaseReader(pth.join(RESULTS_FOLDER_PATH, "cases_octo_prop.sql"))

    solver_case = cr.get_cases(
        "root.performances.solve_equilibrium.compute_dep_equilibrium.nonlinear_solver"
    )
    for case in solver_case:
        print(
            "AC Voltage IN",
            case.outputs[
                "compute_energy_consumed.power_train_performances.motor_4.ac_voltage_peak_in"
            ],
        )
        print(
            "AC Current IN",
            case.outputs[
                "compute_energy_consumed.power_train_performances.motor_4.ac_current_rms_in"
            ],
        )


def test_op_mission_with_target_soc():
    xml_file = "op_mission_inputs.xml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    oad.RegisterSubmodel.active_models["submodel.performances_he.energy_consumption"] = (
        "fastga_he.submodel.performances.energy_consumption.from_pt_file"
    )
    oad.RegisterSubmodel.active_models["submodel.performances_he.dep_effect"] = (
        "fastga_he.submodel.performances.dep_effect.from_pt_file"
    )

    system = OperationalMissionVectorWithTargetSoC(
        number_of_points_climb=30,
        number_of_points_cruise=30,
        number_of_points_descent=20,
        number_of_points_reserve=10,
        power_train_file_path=pt_file_path,
        pre_condition_pt=True,
        use_linesearch=False,
        use_apply_nonlinear=False,
        variable_name_target_SoC="data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min",
    )
    system.nonlinear_solver.options["iprint"] = 2

    ivc = get_indep_var_comp(
        list_inputs(system),
        __file__,
        xml_file,
    )

    problem = run_system(system, ivc)

    # In test_op_mission_vector_from_yml, the mission end with an SoC of 37% so here, since we will
    # have it end at a smaller SoC the range should be greater than the 200 km
    range_array = problem.get_val("data:mission:operational:range", units="km")
    assert range_array == pytest.approx(273, abs=1.0)

    problem.output_file_path = pth.join(RESULTS_FOLDER_PATH, "mission_with_target_SOC.xml")
    problem.write_outputs()


def test_payload_range_elec():
    xml_file = "input_payload_range.xml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    input_list = list_inputs(ComputePayloadRange(power_train_file_path=pt_file_path))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        input_list,
        __file__,
        xml_file,
    )

    problem = run_system(ComputePayloadRange(power_train_file_path=pt_file_path), ivc)
    range_array = problem.get_val("data:mission:payload_range:range", units="NM")
    assert range_array == pytest.approx([0.0, 158, 158, 215], abs=1.0)

    payload_array = problem.get_val("data:mission:payload_range:payload", units="kg")
    assert payload_array == pytest.approx([390.0, 390.0, 390.0, 0.0], abs=1.0)

    ef_array = problem.get_val("data:mission:payload_range:emission_factor") * 1000.0
    assert ef_array == pytest.approx([0.0, 0.29781, 0.29781, 0.0], rel=1e-3)


def test_payload_range_fuel():
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.climb_speed"] = None
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.descent_speed"] = None

    xml_file = "input_payload_range_fuel.xml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "turboshaft_propulsion_for_payload_range.yml")

    input_list = list_inputs(ComputePayloadRange(power_train_file_path=pt_file_path))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        input_list,
        __file__,
        xml_file,
    )

    problem = run_system(ComputePayloadRange(power_train_file_path=pt_file_path), ivc)
    range_array = problem.get_val("data:mission:payload_range:range", units="NM")
    assert range_array == pytest.approx([0.0, 415, 1226, 1350], abs=1.0)

    payload_array = problem.get_val("data:mission:payload_range:payload", units="kg")
    assert payload_array == pytest.approx([1140.0, 1140.0, 578.0, 0.0], abs=1.0)

    ef_array = problem.get_val("data:mission:payload_range:emission_factor") * 1000.0
    assert ef_array == pytest.approx([0.0, 1.7910, 2.8232, 0.0], rel=1e-3)


def test_payload_range_hybrid():
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.climb_speed"] = None
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.descent_speed"] = None

    xml_file = "input_payload_range_hybrid.xml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "hybrid_propulsion.yml")

    input_list = list_inputs(ComputePayloadRange(power_train_file_path=pt_file_path))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        input_list,
        __file__,
        xml_file,
    )

    problem = run_system(ComputePayloadRange(power_train_file_path=pt_file_path), ivc)
    range_array = problem.get_val("data:mission:payload_range:range", units="NM")
    assert range_array == pytest.approx([0.0, 183, 1492, 1586], abs=1.0)

    payload_array = problem.get_val("data:mission:payload_range:payload", units="kg")
    assert payload_array == pytest.approx([1140.0, 1140.0, 367.0, 0.0], abs=1.0)

    ef_array = problem.get_val("data:mission:payload_range:emission_factor") * 1000.0
    assert ef_array == pytest.approx([0.0, 1.9887, 3.6624, 0.0], rel=1e-3)


def test_sample_payload_range_space():
    xml_file = "input_payload_range_hybrid.xml"

    input_list = list_inputs(ComputePayloadRangeInnerSampling(number_of_sample=15))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        input_list,
        __file__,
        xml_file,
    )

    problem = run_system(ComputePayloadRangeInnerSampling(number_of_sample=15), ivc)
    range_array = problem.get_val("data:mission:inner_payload_range:range", units="NM")
    assert range_array == pytest.approx(
        [
            106.0,
            261.0,
            783.0,
            1305.0,
            1567.0,
            106.0,
            259.0,
            778.0,
            1297.0,
            1557.0,
            106.0,
            250.0,
            749.0,
            1249.0,
            1498.0,
            106.0,
            249.0,
            746.0,
            1243.0,
            1492.0,
            106.0,
            191.0,
            574.0,
            957.0,
            1148.0,
            106.0,
            127.0,
            381.0,
            635.0,
            762.0,
            106.0,
            63.0,
            188.0,
            313.0,
            376.0,
            106.0,
            30.0,
            92.0,
            152.0,
            183.0,
        ],
        abs=1.0,
    )

    payload_array = problem.get_val("data:mission:inner_payload_range:payload", units="kg")
    assert payload_array == pytest.approx(
        [
            76.0,
            76.0,
            76.0,
            76.0,
            76.0,
            114.0,
            114.0,
            114.0,
            114.0,
            114.0,
            342.0,
            342.0,
            342.0,
            342.0,
            342.0,
            367.0,
            367.0,
            367.0,
            367.0,
            367.0,
            570.0,
            570.0,
            570.0,
            570.0,
            570.0,
            798.0,
            798.0,
            798.0,
            798.0,
            798.0,
            1026.0,
            1026.0,
            1026.0,
            1026.0,
            1026.0,
            1140.0,
            1140.0,
            1140.0,
            1140.0,
            1140.0,
        ],
        abs=1.0,
    )


def test_payload_range_inner():
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.climb_speed"] = None
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.descent_speed"] = None

    xml_file = "input_payload_range_hybrid.xml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "hybrid_propulsion.yml")

    input_list = list_inputs(ComputePayloadRangeInner(power_train_file_path=pt_file_path))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        input_list,
        __file__,
        xml_file,
    )

    problem = run_system(ComputePayloadRangeInner(power_train_file_path=pt_file_path), ivc)

    fuel_array = problem.get_val("data:mission:inner_payload_range:fuel", units="kg")
    assert fuel_array == pytest.approx(
        [
            131.0,
            215.9,
            509.0,
            814.7,
            973.1,
            131.7,
            215.9,
            508.9,
            814.7,
            972.8,
            135.7,
            217.6,
            509.2,
            814.9,
            972.9,
            139.7,
            189.7,
            419.7,
            658.4,
            780.9,
            143.8,
            156.6,
            313.3,
            474.1,
            556.0,
            148.1,
            121.0,
            200.2,
            280.5,
            321.3,
            150.4,
            101.5,
            141.3,
            180.1,
            200.2,
        ],
        abs=1.0,
    )

    energy_array = problem.get_val("data:mission:inner_payload_range:energy", units="kW*h")
    assert energy_array == pytest.approx(
        [
            0.79,
            1.59,
            6.49,
            13.09,
            16.72,
            1.13,
            2.05,
            7.31,
            13.98,
            17.64,
            4.47,
            6.14,
            12.41,
            19.5,
            23.39,
            9.48,
            10.56,
            15.74,
            21.44,
            24.51,
            14.83,
            15.12,
            18.8,
            22.74,
            24.81,
            20.53,
            19.87,
            21.82,
            23.83,
            24.87,
            23.53,
            22.31,
            23.3,
            24.29,
            24.8,
        ],
        abs=1.0,
    )

    emission_array = problem.get_val("data:mission:inner_payload_range:emissions", units="kg")
    assert emission_array == pytest.approx(
        [
            499.15,
            822.94,
            1940.83,
            3107.4,
            3711.72,
            501.91,
            823.18,
            1940.77,
            3107.65,
            3711.12,
            518.31,
            830.8,
            1943.27,
            3110.03,
            3713.04,
            534.76,
            725.49,
            1603.35,
            2514.06,
            2981.71,
            551.9,
            600.69,
            1198.75,
            1812.19,
            2125.04,
            569.79,
            466.22,
            768.56,
            1074.85,
            1230.77,
            579.02,
            392.58,
            544.56,
            692.59,
            769.44,
        ],
        abs=1.0,
    )
    emission_factor_array = (
        problem.get_val("data:mission:inner_payload_range:emission_factor") * 1000.0
    )
    assert emission_factor_array == pytest.approx(
        [
            33.456,
            22.401,
            17.61,
            16.917,
            16.84,
            22.427,
            15.054,
            11.815,
            11.349,
            11.297,
            7.72,
            5.247,
            4.096,
            3.931,
            3.913,
            4.779,
            3.598,
            2.646,
            2.489,
            2.46,
            3.523,
            3.2,
            2.129,
            1.931,
            1.887,
            2.829,
            3.895,
            2.151,
            1.807,
            1.723,
            2.587,
            6.198,
            2.804,
            2.158,
            1.991,
        ],
        abs=1.0,
    )


def test_payload_range_inner_with_impossible_ranges():
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.climb_speed"] = None
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.descent_speed"] = None

    xml_file = "input_payload_range_fuel.xml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "turboshaft_propulsion_for_payload_range.yml")

    input_list = list_inputs(ComputePayloadRangeInner(power_train_file_path=pt_file_path))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        input_list,
        __file__,
        xml_file,
    )

    problem = run_system(ComputePayloadRangeInner(power_train_file_path=pt_file_path), ivc)
    fuel_array = problem.get_val("data:mission:inner_payload_range:fuel", units="kg")
    assert fuel_array == pytest.approx(
        [
            150.3,
            249.4,
            592.8,
            953.0,
            -1.0,
            151.1,
            249.5,
            592.7,
            952.9,
            -1.0,
            156.3,
            251.9,
            593.0,
            952.6,
            -1.0,
            161.8,
            220.0,
            488.4,
            768.0,
            912.2,
            167.5,
            182.4,
            364.7,
            552.2,
            648.1,
            173.4,
            142.0,
            233.9,
            327.1,
            374.6,
            176.5,
            119.9,
            166.0,
            211.0,
            234.3,
        ],
        abs=1.0,
    )


def test_payload_range_inner_with_builtin_sampling():
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.climb_speed"] = None
    oad.RegisterSubmodel.active_models["submodel.performances.mission_vector.descent_speed"] = None

    xml_file = "input_payload_range_fuel.xml"
    pt_file_path = pth.join(DATA_FOLDER_PATH, "turboshaft_propulsion_for_payload_range.yml")

    input_list = list_inputs(
        ComputePayloadRangeInnerGroup(
            number_of_sample=15, power_train_file_path=pt_file_path, generate_sample=True
        )
    )

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        input_list,
        __file__,
        xml_file,
    )

    problem = run_system(
        ComputePayloadRangeInnerGroup(
            number_of_sample=15, power_train_file_path=pt_file_path, generate_sample=True
        ),
        ivc,
    )

    fuel_array = problem.get_val("data:mission:inner_payload_range:fuel", units="kg")
    assert fuel_array == pytest.approx(
        [
            140.1,
            224.6,
            515.1,
            817.3,
            973.4,
            140.9,
            224.9,
            515.1,
            817.2,
            973.3,
            145.8,
            227.1,
            515.9,
            817.2,
            973.1,
            150.9,
            229.4,
            517.0,
            817.5,
            973.3,
            151.1,
            229.5,
            517.0,
            817.6,
            973.3,
            156.2,
            199.7,
            418.2,
            644.4,
            760.6,
            161.7,
            166.5,
            309.7,
            456.1,
            530.6,
            164.5,
            149.0,
            252.9,
            358.5,
            411.9,
        ],
        abs=1.0,
    )

    energy_array = problem.get_val("data:mission:inner_payload_range:energy", units="kW*h")
    assert energy_array == pytest.approx(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        abs=1.0,
    )

    emission_array = problem.get_val("data:mission:inner_payload_range:emissions", units="kg")
    assert emission_array == pytest.approx(
        [
            533.9,
            855.6,
            1962.3,
            3113.7,
            3708.5,
            536.9,
            856.9,
            1962.7,
            3113.5,
            3708.2,
            555.5,
            865.2,
            1965.6,
            3113.5,
            3707.5,
            574.9,
            873.9,
            1969.7,
            3114.9,
            3708.3,
            575.6,
            874.2,
            1969.9,
            3114.9,
            3708.3,
            595.1,
            760.7,
            1593.5,
            2455.1,
            2897.8,
            616.1,
            634.5,
            1179.8,
            1737.7,
            2021.6,
            626.8,
            567.7,
            963.4,
            1365.8,
            1569.5,
        ],
        abs=1.0,
    )
    emission_factor_array = (
        problem.get_val("data:mission:inner_payload_range:emission_factor") * 1000.0
    )
    assert emission_factor_array == pytest.approx(
        [
            42.145,
            27.348,
            20.907,
            19.904,
            19.756,
            28.257,
            18.372,
            14.026,
            13.35,
            13.25,
            9.745,
            6.42,
            4.862,
            4.621,
            4.585,
            6.051,
            4.046,
            3.04,
            2.884,
            2.861,
            5.975,
            3.997,
            3.002,
            2.848,
            2.826,
            4.474,
            3.399,
            2.374,
            2.194,
            2.158,
            3.602,
            3.457,
            2.143,
            1.894,
            1.836,
            3.299,
            3.887,
            2.199,
            1.871,
            1.791,
        ],
        abs=1.0e-3,
    )

    payload_array = problem.get_val("data:mission:inner_payload_range:payload", units="kg")
    assert payload_array == pytest.approx(
        [
            76.0,
            76.0,
            76.0,
            76.0,
            76.0,
            114.0,
            114.0,
            114.0,
            114.0,
            114.0,
            342.0,
            342.0,
            342.0,
            342.0,
            342.0,
            570.0,
            570.0,
            570.0,
            570.0,
            570.0,
            578.0,
            578.0,
            578.0,
            578.0,
            578.0,
            798.0,
            798.0,
            798.0,
            798.0,
            798.0,
            1026.0,
            1026.0,
            1026.0,
            1026.0,
            1026.0,
            1140.0,
            1140.0,
            1140.0,
            1140.0,
            1140.0,
        ],
        abs=1.0,
    )

    range_array = problem.get_val("data:mission:inner_payload_range:range", units="NM")
    assert range_array == pytest.approx(
        [
            90.0,
            222.3,
            666.8,
            1111.4,
            1333.7,
            90.0,
            220.9,
            662.8,
            1104.6,
            1325.5,
            90.0,
            212.8,
            638.3,
            1063.9,
            1276.6,
            90.0,
            204.6,
            613.9,
            1023.1,
            1227.7,
            90.0,
            204.3,
            613.0,
            1021.7,
            1226.0,
            90.0,
            151.4,
            454.3,
            757.1,
            908.5,
            90.0,
            96.6,
            289.8,
            482.9,
            579.5,
            90.0,
            69.2,
            207.5,
            345.8,
            415.0,
        ],
        abs=1.0,
    )


def test_mission_vector_atr_42():
    # Define used files depending on options
    xml_file_name = "atr42_retrofit.xml"
    process_file_name = "atr42_retrofit.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    #problem.set_val("data:weight:aircraft:MTOW", units="kg", val=15000.0)
    # problem.set_val("data:weight:aircraft:OWE", units="kg", val=10000.0)
    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=10855.697082198)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.114459353773869)
    #problem.set_val("data:weight:aircraft:MZFW", units="kg", val=13000.0)
    #problem.set_val("data:weight:aircraft:MLW", units="kg", val=14000.0)

    # om.n2(problem, show_browser=True)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)