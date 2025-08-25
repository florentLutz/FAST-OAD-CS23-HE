# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import numpy as np
import pytest


import fastoad.api as oad
from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.write_outputs import write_outputs
from utils.filter_residuals import filter_residuals

from .simple_assembly.performances_simple_assembly_splitter import PerformancesAssemblySplitter

from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile
from ..assemblers.sizing_from_pt_file import PowerTrainSizingFromFile
from ..assemblers.delta_from_pt_file import AerodynamicDeltasFromPTFile

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
                111465.98,
                112063.82,
                112651.28,
                113228.31,
                113794.88,
                114350.94,
                114896.45,
                115431.38,
                115955.68,
                116469.32,
            ]
        ),
        abs=1,
    )

    assert problem.get_val("performances.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array(
            [
                4.16,
                4.18,
                4.19,
                4.20,
                4.21,
                4.23,
                4.24,
                4.25,
                4.26,
                4.27,
            ]
        ),
        abs=1e-2,
    )
    # Only one input and one output to the system so input and output should be equal
    assert problem.get_val(
        "performances.fuel_system_1.fuel_consumed_in_t_1", units="kg"
    ) == pytest.approx(
        problem.get_val("performances.ice_1.fuel_consumed_t", units="kg"),
        abs=1e-2,
    )
    assert problem.get_val(
        "performances.fuel_tank_1.fuel_remaining_t", units="kg"
    ) == pytest.approx(
        np.array(
            [
                42.19,
                38.03,
                33.86,
                29.67,
                25.47,
                21.25,
                17.03,
                12.79,
                8.54,
                4.27,
            ]
        ),
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
                88707.55,
                89177.31,
                89638.85,
                90092.16,
                90537.21,
                90973.95,
                91402.37,
                91822.43,
                92234.11,
                92637.38,
            ]
        ),
        abs=1,
    )

    assert problem.get_val("performances.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array([3.65, 3.66, 3.68, 3.7, 3.71, 3.72, 3.73, 3.74, 3.75, 3.76]),
        abs=1e-2,
    )
    # Only one input and one output to the system so input and output should be equal
    assert problem.get_val(
        "performances.fuel_system_1.fuel_consumed_in_t_1", units="kg"
    ) == pytest.approx(
        problem.get_val("performances.ice_1.fuel_consumed_t", units="kg"),
        abs=1e-2,
    )
    assert problem.get_val(
        "performances.fuel_tank_1.fuel_remaining_t", units="kg"
    ) == pytest.approx(
        np.array([37.10, 33.46, 29.79, 26.11, 22.41, 18.7, 14.97, 11.24, 7.51, 3.76]),
        abs=1e-2,
    )

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_performances_splitter_60_40.xml"),
        problem,
    )


def test_assembly_performances_splitter_100_0():
    input_list = list_inputs(PerformancesAssemblySplitter(number_of_points=NB_POINTS_TEST))
    input_list.remove("data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_split")

    ivc = get_indep_var_comp(input_list, __file__, XML_FILE)

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
    ivc.add_output(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_split",
        val=np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        units="percent",
    )

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

    assert voltage_in * current_in == pytest.approx(
        current_out * voltage_out,
        rel=5e-3,
    )
    assert current_in_2 == pytest.approx(np.zeros(NB_POINTS_TEST), abs=5e-3)

    assert voltage_out == pytest.approx(voltage_in_2, rel=5e-3)

    # Only one input and one output to the system so input and output should be equal
    assert problem.get_val(
        "performances.fuel_system_1.fuel_consumed_in_t_1", units="kg"
    ) == pytest.approx(
        problem.get_val("performances.ice_1.fuel_consumed_t", units="kg"),
        abs=1e-2,
    )
    # No fuel is used, so no fuel is loaded so there is no fuel remaining
    assert problem.get_val(
        "performances.fuel_tank_1.fuel_remaining_t", units="kg"
    ) == pytest.approx(np.zeros(NB_POINTS_TEST), abs=1e-2)


def test_assembly_performances_splitter_100_0_only_part():
    input_list = list_inputs(PerformancesAssemblySplitter(number_of_points=NB_POINTS_TEST))
    input_list.remove("data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_split")

    ivc = get_indep_var_comp(input_list, __file__, XML_FILE)
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
    split_array = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 80.0, 70.0])
    ivc.add_output(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_split",
        val=split_array,
        units="percent",
    )

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

    current_out = problem.get_val("performances.dc_splitter_1.dc_current_out", units="A")
    voltage_out = problem.get_val("performances.dc_splitter_1.dc_voltage", units="V")

    current_in = problem.get_val("performances.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("performances.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("performances.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("performances.rectifier_1.dc_voltage_out", units="V")

    assert voltage_in * current_in == pytest.approx(
        current_out * voltage_out * split_array / 100.0,
        rel=5e-3,
    )
    assert current_in_2 == pytest.approx(current_out * (1.0 - split_array / 100.0), abs=5e-3)

    assert voltage_out == pytest.approx(voltage_in_2, rel=5e-3)

    # Only one input and one output to the system so input and output should be equal
    assert problem.get_val(
        "performances.fuel_system_1.fuel_consumed_in_t_1", units="kg"
    ) == pytest.approx(
        problem.get_val("performances.ice_1.fuel_consumed_t", units="kg"),
        abs=1e-2,
    )
    assert problem.get_val(
        "performances.fuel_tank_1.fuel_remaining_t", units="kg"
    ) == pytest.approx(
        np.array([5.70, 5.70, 5.70, 5.70, 5.70, 5.70, 5.70, 5.70, 4.74, 2.85]),
        abs=1e-2,
    )


def test_performances_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_splitter.yml")

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
                111465.98,
                112063.82,
                112651.28,
                113228.31,
                113794.88,
                114350.94,
                114896.45,
                115431.38,
                115955.68,
                116469.32,
            ]
        ),
        rel=1e-3,
    )

    assert problem.get_val("component.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array(
            [
                4.16,
                4.18,
                4.19,
                4.20,
                4.21,
                4.23,
                4.24,
                4.25,
                4.26,
                4.27,
            ]
        ),
        abs=1e-2,
    )
    # Only one input and one output to the system so input and output should be equal
    assert problem.get_val(
        "component.fuel_system_1.fuel_consumed_in_t_1", units="kg"
    ) == pytest.approx(
        problem.get_val("component.ice_1.fuel_consumed_t", units="kg"),
        abs=1e-2,
    )
    assert problem.get_val("component.fuel_tank_1.fuel_remaining_t", units="kg") == pytest.approx(
        np.array(
            [
                42.19,
                38.03,
                33.86,
                29.67,
                25.47,
                21.25,
                17.03,
                12.79,
                8.54,
                4.27,
            ]
        ),
        abs=1e-2,
    )

    # write_outputs(
    #     pth.join(outputs.__path__[0], "simple_assembly_performances_splitter_50_50_pt_file.xml"),
    #     problem,
    # )


def test_assembly_sizing_from_pt_file():
    oad.RegisterSubmodel.active_models["submodel.propulsion.constraints.pmsm.rpm"] = (
        "fastga_he.submodel.propulsion.constraints.pmsm.rpm.enforce"
    )

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
    ) == pytest.approx(19.95, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass", units="kg"
    ) == pytest.approx(22.31, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass", units="kg"
    ) == pytest.approx(266.59, rel=1e-2)
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
    ) == pytest.approx(28.93, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:mass", units="kg"
    ) == pytest.approx(34.08, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:mass", units="kg"
    ) == pytest.approx(0.623, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:mass", units="kg"
    ) == pytest.approx(302.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:mass", units="kg"
    ) == pytest.approx(0.4124, rel=1e-2)

    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        2260.00, rel=1e-2
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
    ivc.add_output("altitude", val=altitude, units="m")
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
