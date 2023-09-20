# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import copy
import os
import os.path as pth

import numpy as np
import pytest

import openmdao.api as om
import fastoad.api as oad
from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.write_outputs import write_outputs
from utils.filter_residuals import filter_residuals

import fastga_he.api as api_he
from fastga_he.powertrain_builder.exceptions import FASTGAHEIncoherentVoltage

from .simple_assembly.performances_simple_assembly_splitter_power_share import (
    PerformancesAssemblySplitterPowerShare,
)
from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile
from ..assemblers.delta_from_pt_file import AerodynamicDeltasFromPTFile

from . import outputs

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
OUTPUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "outputs")

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
        np.array([5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6]),
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
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
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
        np.array([3.72, 3.73, 3.74, 3.75, 3.76, 3.77, 3.78, 3.79, 3.8, 3.81]),
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
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
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
    if not pth.exists(pth.join(OUTPUT_FOLDER_PATH, "cases.sql")):
        recorder = om.SqliteRecorder(pth.join(OUTPUT_FOLDER_PATH, "cases.sql"))
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
        np.array([3.49, 3.9, 4.22, 4.59, 5.03, 5.57, 5.6, 5.6, 5.6, 5.6]),
        abs=1e-2,
    )
    assert problem.get_val(
        "performances.battery_pack_1.state_of_charge", units="percent"
    ) == pytest.approx(
        np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.96, 96.79, 93.42]),
        abs=1e-2,
    )

    problem.check_partials(compact_print=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_case_reader():

    fig = api_he.residuals_viewer(
        recorder_data_file_path=pth.join(OUTPUT_FOLDER_PATH, "cases.sql"),
        case="root.performances.nonlinear_solver",
        power_train_file_path=pth.join(DATA_FOLDER_PATH, "pt_file_equivalent.yml"),
        what_to_plot="residuals",
    )
    fig.show()


def test_assembly_performances_splitter_low_to_high_requirement_from_pt_file():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_splitter_power_share.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
                pre_condition_pt=False,
            )
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
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
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=False,
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
        np.array([3.49, 3.9, 4.22, 4.59, 5.03, 5.57, 5.6, 5.6, 5.6, 5.6]),
        abs=1e-2,
    )
    assert problem.get_val(
        "component.battery_pack_1.state_of_charge", units="percent"
    ) == pytest.approx(
        np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.96, 96.79, 93.42]),
        rel=1e-3,
    )

    problem.check_partials(compact_print=True)


def test_incoherent_voltage():

    # Small test to see if the check that prevent the problem from running with incoherent value
    # works

    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_splitter_power_share.yml")

    input_list = list_inputs(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        )
    )
    input_list.remove(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_out_target_mission"
    )

    ivc_blank = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc_blank.add_output("altitude", val=altitude, units="m")
    ivc_blank.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc_blank.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc_blank.add_output("thrust", val=np.linspace(500, 1500, NB_POINTS_TEST), units="N")
    ivc_blank.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc_blank.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    ivc_work = copy.deepcopy(ivc_blank)

    ivc_work.add_output(
        name="data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_out_target_mission",
        units="V",
        val=np.full(NB_POINTS_TEST, 850),
    )

    # Should work
    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        ),
        ivc_work,
    )

    ivc_workn_t = copy.deepcopy(ivc_blank)
    ivc_workn_t.add_output(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1"
        ":voltage_out_target_mission",
        units="V",
        val=np.full(NB_POINTS_TEST, 400),
    )
    # Should return an exception
    with pytest.raises(FASTGAHEIncoherentVoltage) as e_info:
        problem = run_system(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
                pre_condition_pt=True,
            ),
            ivc_workn_t,
        )

    assert (
        e_info.value.args[0] == "The target voltage chosen for the following input: "
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1"
        ":voltage_out_target_mission, "
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_out_target_mission is "
        "incoherent. Ensure that they have the same value and/or units"
    )


def test_slipstream_from_pt_file():

    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            AerodynamicDeltasFromPTFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
            )
        ),
        __file__,
        XML_FILE,
    )

    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("alpha", val=np.linspace(5.0, 10.0, NB_POINTS_TEST), units="deg")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        AerodynamicDeltasFromPTFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("delta_Cl") * 1e6 == pytest.approx(
        np.array([4097.7, 4246.4, 4382.8, 4507.5, 4621.1, 4724.3, 4817.5, 4901.4, 4976.2, 5042.6]),
        rel=1e-3,
    )
    assert problem.get_val("delta_Cd") * 1e6 == pytest.approx(
        np.array([191.86, 212.4, 233.24, 254.3, 275.49, 296.75, 318.02, 339.23, 360.32, 381.26]),
        rel=1e-3,
    )
    assert problem.get_val("delta_Cm") * 1e9 == pytest.approx(
        np.array(
            [
                -1683.5,
                -1585.9,
                -1494.6,
                -1409.1,
                -1328.9,
                -1253.8,
                -1183.3,
                -1117.1,
                -1054.9,
                -996.6,
            ]
        ),
        rel=1e-3,
    )
