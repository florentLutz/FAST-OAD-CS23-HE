# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import openmdao.api as om
import numpy as np

import pytest

import fastoad.api as oad

from stdatm import Atmosphere

from ..components.propulsor.propeller import PerformancesPropeller
from ..components.source.turboshaft import PerformancesTurboshaft
from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.filter_residuals import filter_residuals

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "simple_turboshaft_assembly.xml"
NB_POINTS_TEST = 25


class PerformancesAssembly(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-8
        self.linear_solver = om.DirectSolver()

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "propeller_1",
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "thrust", "data:*", "density"],
        )
        self.add_subsystem(
            "turboshaft_1",
            PerformancesTurboshaft(
                turboshaft_id="turboshaft_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "density", "true_airspeed", "altitude"],
        )

        self.connect("propeller_1.rpm", "turboshaft_1.rpm")
        self.connect("propeller_1.shaft_power_in", "turboshaft_1.shaft_power_out")


def test_simple_turboshaft_assembly():
    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssembly(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(500, 1500, NB_POINTS_TEST), units="N")
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    problem = oad.FASTOADProblem(reports=False)
    model = problem.model
    model.add_subsystem(
        name="inputs",
        subsys=ivc,
        promotes=["*"],
    )
    model.add_subsystem(
        name="performances",
        subsys=PerformancesAssembly(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val(
        "performances.turboshaft_1.shaft_power_out", units="kW"
    ) == pytest.approx(
        np.array(
            [
                67.0084946,
                70.92851627,
                74.91205887,
                78.95710284,
                83.06216266,
                87.22611799,
                91.4481041,
                95.72743899,
                100.0635737,
                104.45605787,
                108.90451535,
                113.40862671,
                117.96811648,
                122.58274371,
                127.2522949,
                131.9765787,
                136.75542177,
                141.58866564,
                146.47616426,
                151.41778199,
                156.41339214,
                161.46287564,
                166.56612008,
                171.72301887,
                176.93347057,
            ]
        ),
        abs=1e-2,
    )

    assert problem.get_val(
        "performances.turboshaft_1.shaft_power_out", units="kW"
    ) == pytest.approx(
        problem.get_val("performances.propeller_1.shaft_power_in", units="kW"), rel=1e-4
    )

    assert problem.get_val(
        "performances.turboshaft_1.fuel_consumed_t", units="kg"
    ) == pytest.approx(
        np.array(
            [
                18.163,
                18.450,
                18.739,
                19.028,
                19.318,
                19.609,
                19.900,
                20.192,
                20.484,
                20.777,
                21.071,
                21.365,
                21.660,
                21.955,
                22.250,
                22.546,
                22.843,
                23.140,
                23.438,
                23.736,
                24.035,
                24.335,
                24.634,
                24.935,
                25.236,
            ]
        ),
        abs=1e-2,
    )


def test_assembly_via_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_turboshaft_assembly_test.yml")

    powertrain_performance = PowerTrainPerformancesFromFile(
        power_train_file_path=pt_file_path,
        number_of_points=NB_POINTS_TEST,
        pre_condition_pt=True,
    )

    powertrain_performance.configurator._cache["skip_test"] = True

    ivc = get_indep_var_comp(
        list_inputs(powertrain_performance),
        __file__,
        XML_FILE,
    )

    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(500, 1500, NB_POINTS_TEST), units="N")
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    problem = run_system(
        powertrain_performance,
        ivc,
    )

    assert problem.get_val("component.turboshaft_1.shaft_power_out", units="kW") == pytest.approx(
        np.array(
            [
                67.0084946,
                70.92851627,
                74.91205887,
                78.95710284,
                83.06216266,
                87.22611799,
                91.4481041,
                95.72743899,
                100.0635737,
                104.45605787,
                108.90451535,
                113.40862671,
                117.96811648,
                122.58274371,
                127.2522949,
                131.9765787,
                136.75542177,
                141.58866564,
                146.47616426,
                151.41778199,
                156.41339214,
                161.46287564,
                166.56612008,
                171.72301887,
                176.93347057,
            ]
        ),
        abs=1e-2,
    )

    assert problem.get_val("component.turboshaft_1.shaft_power_out", units="kW") == pytest.approx(
        problem.get_val("component.propeller_1.shaft_power_in", units="kW"), rel=1e-4
    )

    assert problem.get_val("component.turboshaft_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array(
            [
                18.163,
                18.450,
                18.739,
                19.028,
                19.318,
                19.609,
                19.900,
                20.192,
                20.484,
                20.777,
                21.071,
                21.365,
                21.660,
                21.955,
                22.250,
                22.546,
                22.843,
                23.140,
                23.438,
                23.736,
                24.035,
                24.335,
                24.634,
                24.935,
                25.236,
            ]
        ),
        abs=1e-2,
    )

    powertrain_performance.configurator._cache["skip_test"] = False
