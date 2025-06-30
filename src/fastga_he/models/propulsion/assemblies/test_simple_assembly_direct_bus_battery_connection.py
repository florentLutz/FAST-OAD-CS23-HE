# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import copy
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
from ..components.connectors.dc_cable.constants import (
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE,
)

from . import outputs

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
OUTPUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "outputs")

XML_FILE = "simple_assembly_direct_bus_battery_connection.xml"
NB_POINTS_TEST = 10


@pytest.fixture()
def restore_submodels():
    """
    Since the submodels in the configuration file differ from the defaults, this restore process
    ensures subsequent assembly tests run under default conditions.
    """
    old_submodels = copy.deepcopy(oad.RegisterSubmodel.active_models)
    yield
    oad.RegisterSubmodel.active_models = old_submodels


def test_assembly_performances(restore_submodels):
    oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE] = (
        "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.steady_state"
    )
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
                192255.0,
                193250.0,
                194227.0,
                195186.0,
                196127.0,
                197051.0,
                197958.0,
                198847.0,
                199720.0,
                200575.0,
            ]
        ),
        abs=1,
    )

    # We check that it decreases with time, because as time goes on, SOC decrease so R_int increases
    assert problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                895.39620638,
                885.38155954,
                877.0304345,
                869.32945753,
                861.51687284,
                853.09235584,
                843.80317964,
                833.61692948,
                822.68562866,
                811.30075336,
            ]
        ),
        abs=1,
    )


def test_assembly_performances_constant_demand(restore_submodels):
    # We now check with a constant power demand, this way we will see that the dc bus 2 voltage
    # still drops and that the efficiency of the inverter varies.
    oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE] = (
        "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.steady_state"
    )

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
                872.0,
                865.0,
                857.0,
                849.0,
                840.0,
                831.0,
                821.0,
            ]
        ),
        abs=1,
    )
    # Almost imperceptible
    assert problem.get_val("performances.inverter_1.efficiency") == pytest.approx(
        np.array(
            [
                0.97301568,
                0.97301703,
                0.97301817,
                0.97301923,
                0.97302031,
                0.97302147,
                0.97302275,
                0.97302417,
                0.97302574,
                0.97302743,
            ]
        ),
        abs=1e-5,
    )

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2_direct.html"))


def test_performances_from_pt_file(restore_submodels):
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_direct_bus_battery_connection.yml")
    oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE] = (
        "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.steady_state"
    )

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
                192255.21282562,
                193249.9663507,
                194226.77293674,
                195185.56587769,
                196126.51139977,
                197049.94417756,
                197956.38133897,
                198846.43060304,
                199720.24266326,
                200576.28667607,
            ]
        ),
        rel=1e-4,
    )

    # We check that it decreases with time, because as time goes on, SOC decrease so R_int increases
    assert problem.get_val("component.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                895.39621349,
                885.38160954,
                877.03048832,
                869.32956305,
                861.51712019,
                853.09282268,
                843.80388163,
                833.61776084,
                822.68632196,
                811.30097904,
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


def test_assembly_performances_with_direct_sspc_battery_connection(restore_submodels):
    oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE] = (
        "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.steady_state"
    )

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
                192258.0,
                193253.0,
                194230.0,
                195189.0,
                196131.0,
                197055.0,
                197962.0,
                198852.0,
                199724.0,
                200580.0,
            ]
        ),
        abs=1,
    )

    assert problem.get_val(
        "performances.battery_pack_1.dc_current_out", units="A"
    ) * problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                194200.49888407,
                195205.61889228,
                196192.45014373,
                197161.29475812,
                198112.41214235,
                199045.9949277,
                199962.1590302,
                200860.93954574,
                201742.28824555,
                202606.07385605,
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
                895.28347477,
                885.15148112,
                876.71570352,
                868.93271137,
                861.02173362,
                852.47445054,
                843.03983939,
                832.69492961,
                821.60649911,
                810.08291774,
            ]
        ),
        abs=1,
    )


def test_direct_sspc_battery_connection_from_pt_file(restore_submodels):
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_direct_sspc_battery_connection.yml")
    oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE] = (
        "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.steady_state"
    )

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
                192258.37459708,
                193253.29806419,
                194230.25143033,
                195189.19351206,
                196130.3109528,
                197053.95347946,
                197960.64717498,
                198850.98275428,
                199725.03824276,
                200581.1924893,
            ]
        ),
        rel=1e-4,
    )

    assert problem.get_val("component.battery_pack_1.dc_current_out", units="A") * problem.get_val(
        "component.battery_pack_1.voltage_out", units="V"
    ) == pytest.approx(
        np.array(
            [
                194200.37838089,
                195205.35157999,
                196192.17316195,
                197160.80152733,
                198111.42520484,
                199044.397454,
                199960.24967169,
                200859.57853967,
                201742.46287148,
                202607.26514071,
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
                895.28348175,
                885.15152969,
                876.7157557,
                868.93281485,
                861.02197634,
                852.47490646,
                843.04051915,
                832.6957218,
                821.6071355,
                810.08308563,
            ]
        ),
        rel=1e-4,
    )
