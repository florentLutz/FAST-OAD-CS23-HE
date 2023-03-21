# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

import fastoad.api as oad
from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.write_outputs import write_outputs

from .simple_assembly.performances_simple_assembly_splitter import PerformancesAssemblySplitter

from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile
from ..assemblers.sizing_from_pt_file import PowerTrainSizingFromFile

from . import outputs

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "simple_assembly_splitter.xml"
NB_POINTS_TEST = 10


def test_assembly_performances_splitter_50_50():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssemblySplitter(number_of_points=NB_POINTS_TEST)),
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
        subsys=PerformancesAssemblySplitter(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()

    current_in = problem.get_val("performances.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("performances.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("performances.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("performances.rectifier_1.dc_voltage_out", units="V")

    assert current_in * voltage_in == pytest.approx(
        current_in_2 * voltage_in_2,
        abs=1,
    )

    torque_generator = problem.get_val("performances.generator_1.torque_in", units="N*m")
    omega_generator = (
        problem.get_val("performances.generator_1.rpm", units="min**-1") * 2.0 * np.pi / 60.0
    )

    assert torque_generator * omega_generator == pytest.approx(
        np.array(
            [
                99500.7,
                100002.5,
                100496.2,
                100981.6,
                101458.6,
                101927.2,
                102387.3,
                102838.7,
                103281.4,
                103715.3,
            ]
        ),
        abs=1,
    )

    assert problem.get_val("performances.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([3.9, 3.91, 3.92, 3.94, 3.95, 3.96, 3.97, 3.97, 3.98, 3.99]),
        abs=1e-2,
    )

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_performances_splitter_50_50.xml"),
        problem,
    )


def test_assembly_performances_splitter_60_40():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssemblySplitter(number_of_points=NB_POINTS_TEST)),
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
        subsys=PerformancesAssemblySplitter(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_split",
        val=60.0,
        units="percent",
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()

    current_in = problem.get_val("performances.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("performances.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("performances.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("performances.rectifier_1.dc_voltage_out", units="V")

    assert voltage_in * current_in == pytest.approx(
        1.5 * voltage_in_2 * current_in_2,
        rel=5e-3,
    )

    torque_generator = problem.get_val("performances.generator_1.torque_in", units="N*m")
    omega_generator = (
        problem.get_val("performances.generator_1.rpm", units="min**-1") * 2.0 * np.pi / 60.0
    )

    assert torque_generator * omega_generator == pytest.approx(
        np.array(
            [
                79295.1,
                79690.3,
                80079.0,
                80461.2,
                80836.7,
                81205.6,
                81567.7,
                81923.0,
                82271.4,
                82612.8,
            ]
        ),
        abs=1,
    )

    assert problem.get_val("performances.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([3.26, 3.27, 3.29, 3.31, 3.32, 3.34, 3.35, 3.37, 3.38, 3.4]),
        abs=1e-2,
    )

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_performances_splitter_60_40.xml"),
        problem,
    )


def test_performances_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_splitter.yml")

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
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
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

    current_in = problem.get_val("component.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("component.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("component.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("component.rectifier_1.dc_voltage_out", units="V")

    assert current_in * voltage_in == pytest.approx(
        current_in_2 * voltage_in_2,
        abs=1,
    )

    torque_generator = problem.get_val("component.generator_1.torque_in", units="N*m")
    omega_generator = (
        problem.get_val("component.generator_1.rpm", units="min**-1") * 2.0 * np.pi / 60.0
    )

    assert torque_generator * omega_generator == pytest.approx(
        np.array(
            [
                99500.7,
                100002.5,
                100496.2,
                100981.6,
                101458.6,
                101927.2,
                102387.3,
                102838.7,
                103281.4,
                103715.3,
            ]
        ),
        rel=1e-3,
    )

    assert problem.get_val("component.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([3.9, 3.91, 3.92, 3.94, 3.95, 3.96, 3.97, 3.97, 3.98, 3.99]),
        abs=1e-2,
    )

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_performances_splitter_50_50_pt_file.xml"),
        problem,
    )


def test_assembly_sizing_from_pt_file():

    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.pmsm.rpm"
    ] = "fastga_he.submodel.propulsion.constraints.pmsm.rpm.enforce"

    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_splitter.yml")

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainSizingFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(PowerTrainSizingFromFile(power_train_file_path=pt_file_path), ivc)
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:mass", units="kg"
    ) == pytest.approx(36.35, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(16.19, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:mass", units="kg"
    ) == pytest.approx(11.50, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass", units="kg"
    ) == pytest.approx(22.31, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass", units="kg"
    ) == pytest.approx(96.63, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(1500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:mass", units="kg"
    ) == pytest.approx(6.40, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_2:mass", units="kg"
    ) == pytest.approx(6.40, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_3:mass", units="kg"
    ) == pytest.approx(6.40, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:mass", units="kg"
    ) == pytest.approx(5.68, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:mass", units="kg"
    ) == pytest.approx(34.08, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:mass", units="kg"
    ) == pytest.approx(0.623, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:mass", units="kg"
    ) == pytest.approx(353.50, rel=1e-2)

    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        2097.11, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:CG:x", units="m") == pytest.approx(
        2.847, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:low_speed:CD0") == pytest.approx(
        0.00567915, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:cruise:CD0") == pytest.approx(
        0.00561219, rel=1e-2
    )

    write_outputs(
        pth.join(outputs.__path__[0], "assembly_sizing_from_pt_file.xml"),
        problem,
    )
