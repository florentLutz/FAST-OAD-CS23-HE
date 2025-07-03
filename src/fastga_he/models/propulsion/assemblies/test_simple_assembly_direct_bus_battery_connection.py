# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import numpy as np
import pytest

import openmdao.api as om
import fastoad.api as oad
from stdatm import Atmosphere

import plotly.graph_objects as go

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.filter_residuals import filter_residuals
from utils.write_outputs import write_outputs

from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile

from .simple_assembly.performances_simple_assembly_direct_bus_battery_connection import (
    PerformancesAssemblyDirectBusBatteryConnection,
)
from .simple_assembly.performances_simple_assembly_direct_sspc_battery_connection import (
    PerformancesAssemblyDirectSSPCBatteryConnection,
)

from . import outputs

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
OUTPUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "outputs")

XML_FILE = "simple_assembly_direct_bus_battery_connection.xml"
NB_POINTS_TEST = 10


def test_assembly_performances():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesAssemblyDirectBusBatteryConnection(number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
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
        subsys=PerformancesAssemblyDirectBusBatteryConnection(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()

    # Adding a recorder
    recorder = om.SqliteRecorder(pth.join(OUTPUT_FOLDER_PATH, "cases.sql"))
    solver = model.performances.nonlinear_solver
    solver.add_recorder(recorder)
    solver.recording_options["record_solver_residuals"] = True

    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()

    write_outputs(
        pth.join(
            outputs.__path__[0], "simple_assembly_direct_bus_battery_connection_performances.xml"
        ),
        problem,
    )

    assert problem.get_val(
        "performances.battery_pack_1.dc_current_out", units="A"
    ) * problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                198713.5,
                199751.8,
                200771.0,
                201771.7,
                202754.1,
                203718.4,
                204664.6,
                205592.9,
                206503.2,
                207395.2,
            ]
        ),
        abs=1,
    )

    # We check that it decreases with time, because as time goes on, SOC decrease so R_int increases
    assert problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                895.0,
                884.6,
                875.98,
                868.0,
                859.9,
                851.0,
                841.2,
                830.5,
                819.1,
                807.2,
            ]
        ),
        abs=1,
    )


def test_assembly_performances_constant_demand():
    # We now check with a constant power demand, this way we will see that the dc bus 2 voltage
    # still drops and that the efficiency of the inverter varies.
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesAssemblyDirectBusBatteryConnection(number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.full(NB_POINTS_TEST, 81.8), units="m/s")
    ivc.add_output("thrust", val=np.full(NB_POINTS_TEST, 1450), units="N")
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
        subsys=PerformancesAssemblyDirectBusBatteryConnection(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()

    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    assert problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                896.0,
                886.0,
                879.0,
                871.0,
                863.7,
                855.9,
                847.5,
                838.2,
                828.3,
                817.9,
            ]
        ),
        abs=1,
    )

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2_direct.html"))


def test_performances_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_direct_bus_battery_connection.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
                pre_condition_pt=True,
            )
        ),
        __file__,
        XML_FILE,
    )

    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        ),
        ivc,
    )

    # om.n2(problem)

    _, _, residuals = problem.model.component.get_nonlinear_vectors()

    assert problem.get_val("component.battery_pack_1.dc_current_out", units="A") * problem.get_val(
        "component.battery_pack_1.voltage_out", units="V"
    ) == pytest.approx(
        np.array(
            [
                198713.5,
                199751.8,
                200771.0,
                201771.7,
                202754.1,
                203718.4,
                204664.6,
                205592.9,
                206503.2,
                207395.2,
            ]
        ),
        rel=1e-4,
    )

    # We check that it decreases with time, because as time goes on, SOC decrease so R_int increases
    assert problem.get_val("component.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                895.0,
                884.6,
                875.98,
                868.0,
                859.9,
                851.0,
                841.2,
                830.5,
                819.1,
                807.2,
            ]
        ),
        abs=1,
    )


def test_read_recording():
    recorder_data_file_path = pth.join(OUTPUT_FOLDER_PATH, "cases.sql")

    cr = om.CaseReader(recorder_data_file_path)

    cases = cr.get_cases("root.performances.nonlinear_solver")

    fig = go.Figure()

    y_axis = []

    for case in cases:
        y_axis.append(np.mean(case.residuals["dc_bus_2.dc_voltage"]))

    scatter = go.Scatter(y=y_axis)
    fig.add_trace(scatter)
    fig.show()


def test_assembly_performances_with_direct_sspc_battery_connection():
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesAssemblyDirectSSPCBatteryConnection(number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
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
        subsys=PerformancesAssemblyDirectSSPCBatteryConnection(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()

    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    write_outputs(
        pth.join(
            outputs.__path__[0], "simple_assembly_direct_sspc_battery_connection_performances.xml"
        ),
        problem,
    )

    assert problem.get_val("performances.dc_sspc_3.dc_current_in", units="A") * problem.get_val(
        "performances.dc_sspc_3.dc_voltage_in", units="V"
    ) == pytest.approx(
        np.array(
            [
                198716.9,
                199755.3,
                200774.7,
                201775.5,
                202758.0,
                203722.5,
                204669.0,
                205597.5,
                206508.0,
                207400.4,
            ]
        ),
        abs=1,
    )

    assert problem.get_val(
        "performances.battery_pack_1.dc_current_out", units="A"
    ) * problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                200724.1,
                201773.0,
                202802.7,
                203813.6,
                204806.1,
                205780.3,
                206736.3,
                207674.3,
                208594.0,
                209495.3,
            ]
        ),
        abs=1,
    )

    # We check that it decreases with time, because as time goes on, SOC decrease so R_int
    # increases. Almost no difference with the case where the battery is directly linked with a
    # bus because the battery imposes the voltage so if we were to check on the inverter we would
    # see something significantly lower.
    assert problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                895.3,
                885.2,
                875.7,
                867.6,
                859.3,
                850.4,
                840.4,
                829.6,
                817.9,
                806.0,
            ]
        ),
        abs=1,
    )


def test_direct_sspc_battery_connection_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_direct_sspc_battery_connection.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
                pre_condition_pt=True,
            )
        ),
        __file__,
        XML_FILE,
    )

    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        ),
        ivc,
    )

    # om.n2(problem)

    _, _, residuals = problem.model.component.get_nonlinear_vectors()

    assert problem.get_val("component.dc_sspc_3.dc_current_in", units="A") * problem.get_val(
        "component.dc_sspc_3.dc_voltage_in", units="V"
    ) == pytest.approx(
        np.array(
            [
                198716.9,
                199755.3,
                200774.7,
                201775.5,
                202758.0,
                203722.5,
                204669.0,
                205597.5,
                206508.0,
                207400.4,
            ]
        ),
        rel=1e-4,
    )

    assert problem.get_val("component.battery_pack_1.dc_current_out", units="A") * problem.get_val(
        "component.battery_pack_1.voltage_out", units="V"
    ) == pytest.approx(
        np.array(
            [
                200724.1,
                201773.0,
                202802.7,
                203813.6,
                204806.1,
                205780.3,
                206736.3,
                207674.3,
                208594.0,
                209495.3,
            ]
        ),
        rel=1e-4,
    )

    # We check that it decreases with time, because as time goes on, SOC decrease so R_int
    # increases. Almost no difference with the case where the battery is directly linked with a
    # bus because the battery imposes the voltage so if we were to check on the inverter we would
    # see something significantly lower.
    assert problem.get_val("component.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                894.9,
                884.4,
                875.7,
                867.6,
                859.3,
                850.4,
                840.4,
                829.6,
                817.9,
                806.0,
            ]
        ),
        rel=1e-4,
    )
