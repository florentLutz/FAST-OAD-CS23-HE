"""
Test mission vector module.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import numpy as np
import openmdao.api as om

import plotly.graph_objects as go

import fastoad.api as oad

import fastga_he.api as oad_he

from fastga_he.models.performances.mission_vector.initialization.initialize_altitude import (
    InitializeAltitude,
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
from fastga_he.models.performances.mission_vector.initialization.initialize_gamma import (
    InitializeGamma,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_airspeed_derivatives import (
    InitializeAirspeedDerivatives,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_time_and_distance import (
    InitializeTimeAndDistance,
)
from fastga_he.models.performances.mission_vector.initialization.initialize_cg import InitializeCoG
from fastga_he.models.performances.mission_vector.mission_vector import MissionVector
from fastga_he.models.propulsion.assemblers.sizing_from_pt_file import PowerTrainSizingFromFile

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
XML_FILE = "sample_ac.xml"
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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

    problem.check_partials(compact_print=False)


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
            0.127,
            0.098,
            0.088,
            0.094,
            0.114,
            0.023,
            0.073,
            0.021,
            0.092,
            0.070,
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
            -0.079,
            -0.0542,
            -0.0138,
            -0.0560,
            -0.0713,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert np.max(np.abs(problem.get_val("d_vx_dt", units="m/s**2") - expected_derivative)) <= 1e-1

    problem.check_partials(compact_print=True)


def test_initialize_gamma():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            InitializeGamma(
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

    problem = run_system(
        InitializeGamma(
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
    assert (
        np.max(np.abs(problem.get_val("vertical_speed", units="m/s") - expected_vertical_speed))
        <= 1e-1
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
    assert np.max(np.abs(problem.get_val("gamma", units="rad") - expected_gamma)) <= 1e-1

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
    ivc.add_output("fuel_consumed_t", units="kg", val=np.linspace(1.0, 2.0, 30))

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
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
            3.478,
        ]
    )
    assert problem.get_val("x_cg", units="m") == pytest.approx(expected_cog, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_mission_vector():

    # Research independent input value in .xml file
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
    assert sizing_fuel == pytest.approx(45.57, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)


def test_mission_vector_from_yml():

    # Define used files depending on options
    xml_file_name = "sample_ac.xml"
    process_file_name = "mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
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
    assert sizing_energy == pytest.approx(150.03, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(0.055, abs=1e-2)


def test_mission_vector_from_yml_fuel():

    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "fuel_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(34.30, abs=1e-2)
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
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
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


def test_mission_vector_from_yml_fuel_turbo():

    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "fuel_turbo_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(39.18, rel=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(0.0, abs=1e-2)


def test_mission_vector_from_yml_fuel_and_battery():

    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"
    process_file_name = "fuel_and_battery_propulsion_mission_vector.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(20.18, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(74.816, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.056, abs=1e-2)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_recording():

    oad.RegisterSubmodel.active_models[
        "submodel.performances_he.energy_consumption"
    ] = "fastga_he.submodel.performances.energy_consumption.from_pt_file"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.pmsm.rpm"
    ] = "fastga_he.submodel.propulsion.constraints.pmsm.rpm.ensure"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.battery.state_of_charge"
    ] = "fastga_he.submodel.propulsion.constraints.battery.state_of_charge.enforce"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.performances.dc_line.temperature_profile"
    ] = "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.with_dynamics"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.inverter.current"
    ] = "fastga_he.submodel.propulsion.constraints.inverter.current.enforce"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.pmsm.torque"
    ] = "fastga_he.submodel.propulsion.constraints.pmsm.torque.enforce"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.generator.rpm"
    ] = "fastga_he.submodel.propulsion.constraints.generator.rpm.ensure"

    # Define used files depending on options
    xml_file_name = "sample_ac_fuel_and_battery_propulsion.xml"

    problem = om.Problem()
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

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(20.18, abs=1e-2)


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
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    sizing_fuel = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert sizing_fuel == pytest.approx(25.88, abs=1e-2)
    sizing_energy = problem.get_val("data:mission:sizing:energy", units="kW*h")
    assert sizing_energy == pytest.approx(50.303, abs=1e-2)
    mission_end_soc = problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    )
    assert mission_end_soc == pytest.approx(-0.271, abs=1e-2)
