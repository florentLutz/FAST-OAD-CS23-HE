# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import numpy as np
import pytest

import openmdao.api as om
import fastoad.api as oad
from stdatm import Atmosphere

import plotly.graph_objects as go

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.write_outputs import write_outputs
from utils.filter_residuals import filter_residuals

from .simple_assembly.performances_simple_assembly_splitter_power_share import (
    PerformancesAssemblySplitterPowerShare,
)
from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile


from . import outputs

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "simple_assembly_splitter.xml"
NB_POINTS_TEST = 10


def test_assembly_performances_splitter_150_kw():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssemblySplitterPowerShare(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
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
        subsys=PerformancesAssemblySplitterPowerShare(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    current_out = problem.get_val("performances.dc_splitter_1.dc_current_out", units="A")
    voltage_out = problem.get_val("performances.dc_splitter_1.dc_voltage", units="V")

    current_in = problem.get_val("performances.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("performances.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("performances.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("performances.rectifier_1.dc_voltage_out", units="V")

    assert current_in * voltage_in == pytest.approx(
        current_out * voltage_out - np.full(NB_POINTS_TEST, 150.0e3),
        abs=1,
    )

    assert current_in_2 * voltage_in_2 == pytest.approx(
        np.full(NB_POINTS_TEST, 150.0e3),
        abs=1,
    )

    assert problem.get_val("performances.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([5.58, 5.58, 5.58, 5.58, 5.58, 5.58, 5.58, 5.58, 5.58, 5.58]),
        abs=1e-2,
    )

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_performances_splitter_power_share.xml"),
        problem,
    )

    # problem.check_partials(compact_print=True)


def test_assembly_performances_splitter_150_kw_low_requirement():

    # Same test as above except the thrust required will be much lower to check if it indeed
    # output zero current in the secondary branch and primary branch is equal to the output

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssemblySplitterPowerShare(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(550, 500, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
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
        subsys=PerformancesAssemblySplitterPowerShare(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    current_out = problem.get_val("performances.dc_splitter_1.dc_current_out", units="A")
    voltage_out = problem.get_val("performances.dc_splitter_1.dc_voltage", units="V")

    current_in = problem.get_val("performances.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("performances.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("performances.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("performances.rectifier_1.dc_voltage_out", units="V")

    assert current_in * voltage_in == pytest.approx(
        np.zeros(NB_POINTS_TEST),
        abs=1,
    )

    assert current_in_2 * voltage_in_2 == pytest.approx(
        current_out * voltage_out,
        abs=1,
    )

    assert problem.get_val("performances.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([3.54, 3.56, 3.58, 3.6, 3.62, 3.63, 3.65, 3.67, 3.69, 3.7]),
        abs=1e-2,
    )

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_assembly_performances_splitter_150_kw_low_to_high_requirement():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssemblySplitterPowerShare(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(500, 1500, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
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
        subsys=PerformancesAssemblySplitterPowerShare(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()
    # Run problem and check obtained value(s) is/(are) correct

    # Adding a recorder
    if not pth.exists("propulsion/assemblies/outputs/cases.sql"):
        recorder = om.SqliteRecorder("propulsion/assemblies/outputs/cases.sql")
        solver = model.performances.nonlinear_solver
        solver.add_recorder(recorder)
        solver.recording_options["record_solver_residuals"] = True

    problem.run_model()

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    current_out = problem.get_val("performances.dc_splitter_1.dc_current_out", units="A")
    voltage_out = problem.get_val("performances.dc_splitter_1.dc_voltage", units="V")

    current_in = problem.get_val("performances.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("performances.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("performances.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("performances.rectifier_1.dc_voltage_out", units="V")

    assert current_out * voltage_out == pytest.approx(
        current_in * voltage_in + current_in_2 * voltage_in_2, rel=5e-3
    )

    assert problem.get_val("performances.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([3.32, 3.79, 4.09, 4.42, 4.81, 5.27, 5.58, 5.58, 5.58, 5.58]),
        abs=1e-2,
    )
    assert problem.get_val(
        "performances.battery_pack_1.state_of_charge", units="percent"
    ) == pytest.approx(
        np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.54, 98.04, 95.42]),
        abs=1e-2,
    )

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_case_reader():

    fig = go.Figure()
    cr = om.CaseReader("propulsion/assemblies/outputs/cases.sql")

    solver_case = cr.get_cases("root.performances.nonlinear_solver")
    splitter_voltage_mean = []
    splitter_voltage_min = []
    splitter_voltage_max = []
    for i, case in enumerate(solver_case):

        splitter_voltage_mean.append(np.mean(case.residuals["dc_bus_1.dc_voltage"]))
        splitter_voltage_min.append(np.min(case.residuals["dc_bus_1.dc_voltage"]))
        splitter_voltage_max.append(np.max(case.residuals["dc_bus_1.dc_voltage"]))

    scatter_mean = go.Scatter(
        x=np.arange(len(splitter_voltage_mean)),
        y=splitter_voltage_mean,
        mode="lines+markers",
        name="Mean splitter voltage residuals array",
    )
    fig.add_trace(scatter_mean)
    scatter_min = go.Scatter(
        x=np.arange(len(splitter_voltage_mean)),
        y=splitter_voltage_min,
        mode="lines+markers",
        name="Min splitter voltage residuals array",
    )
    fig.add_trace(scatter_min)
    scatter_max = go.Scatter(
        x=np.arange(len(splitter_voltage_mean)),
        y=splitter_voltage_max,
        mode="lines+markers",
        name="Max splitter voltage residuals array",
    )
    fig.add_trace(scatter_max)

    fig.show()


def test_assembly_performances_splitter_low_to_high_requirement_from_pt_file():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_splitter_power_share.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST
            )
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(500, 1500, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    # om.n2(problem)

    _, _, residuals = problem.model.component.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    current_out = problem.get_val("component.dc_splitter_1.dc_current_out", units="A")
    voltage_out = problem.get_val("component.dc_splitter_1.dc_voltage", units="V")

    current_in = problem.get_val("component.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("component.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("component.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("component.rectifier_1.dc_voltage_out", units="V")

    assert current_out * voltage_out == pytest.approx(
        current_in * voltage_in + current_in_2 * voltage_in_2, rel=5e-3
    )

    assert problem.get_val("component.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([3.32, 3.79, 4.09, 4.42, 4.81, 5.27, 5.58, 5.58, 5.58, 5.58]),
        abs=1e-2,
    )
    assert problem.get_val(
        "component.battery_pack_1.state_of_charge", units="percent"
    ) == pytest.approx(
        np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.54, 98.04, 95.42]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)
