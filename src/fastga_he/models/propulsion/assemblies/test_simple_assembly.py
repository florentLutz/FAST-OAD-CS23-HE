# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

import fastoad.api as oad
import openmdao.api as om
from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.write_outputs import write_outputs

from .simple_assembly.performances_simple_assembly import PerformancesAssembly
from .simple_assembly.sizing_simple_assembly import SizingAssembly
from .simple_assembly.full_simple_assembly import FullSimpleAssembly

from ..assemblers.sizing_from_pt_file import PowerTrainSizingFromFile
from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile
from ..assemblers.mass_from_pt_file import PowerTrainMassFromFile
from ..assemblers.cg_from_pt_file import PowerTrainCGFromFile
from ..assemblers.drag_from_pt_file import PowerTrainDragFromFile

from . import outputs

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "simple_assembly.xml"
NB_POINTS_TEST = 10


def test_assembly_performances():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssembly(number_of_points=NB_POINTS_TEST)),
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
        subsys=PerformancesAssembly(number_of_points=NB_POINTS_TEST),
        promotes=["*"],
    )
    problem.setup()
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    print("\n=========== Propulsive power ===========")
    print(problem.get_val("true_airspeed", units="m/s") * problem.get_val("thrust", units="N"))

    print("\n=========== Shaft power ===========")
    print(problem.get_val("performances.propeller_1.shaft_power_in", units="W"))

    print("\n=========== AC power ===========")
    print(
        problem.get_val("performances.motor_1.ac_current_rms_in", units="A")
        * problem.get_val("performances.motor_1.ac_voltage_rms_in", units="V")
    )

    print("\n=========== DC power before inverter ===========")
    print(
        problem.get_val("performances.inverter_1.dc_current_in", units="A")
        * problem.get_val("performances.inverter_1.dc_voltage_in", units="V")
    )

    print("\n=========== DC power before bus/end of cable ===========")
    print(
        problem.get_val("performances.dc_line_1.dc_current", units="A")
        * problem.get_val("performances.dc_line_1.dc_voltage_out", units="V")
    )

    print("\n=========== DC power start of cable ===========")
    print(
        problem.get_val("performances.dc_line_1.dc_current", units="A")
        * problem.get_val("performances.dc_line_1.dc_voltage_in", units="V")
    )
    # Result is not really accurate since we used a ICE propeller coupled to a small PMSM not
    # sized for the demand, though it shows that the assembly works just fine

    print("\n=========== DC power input of the DC/DC converter ===========")
    print(
        problem.get_val("performances.dc_dc_converter_1.dc_current_in", units="A")
        * problem.get_val("performances.dc_dc_converter_1.dc_voltage_in", units="V")
    )

    print("\n=========== Battery SOC ===========")
    print(problem.get_val("performances.battery_pack_1.state_of_charge", units="percent"))

    print("\n=========== Battery losses ===========")
    print(problem.get_val("performances.battery_pack_1.battery_losses.losses_battery", units="W"))

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()

    assert problem.get_val(
        "performances.dc_dc_converter_1.dc_current_in", units="A"
    ) * problem.get_val("performances.dc_dc_converter_1.dc_voltage_in", units="V") == pytest.approx(
        np.array(
            [
                187277.0,
                188236.0,
                189173.0,
                190095.0,
                191006.0,
                191907.0,
                192800.0,
                193680.0,
                194546.0,
                195395.0,
            ]
        ),
        abs=1,
    )

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_performances.xml"),
        problem,
    )


def test_assembly_sizing():

    ivc = get_indep_var_comp(list_inputs(SizingAssembly()), __file__, XML_FILE)

    problem = oad.FASTOADProblem(reports=False)
    model = problem.model
    model.add_subsystem(name="inputs", subsys=ivc, promotes=["*"])
    model.add_subsystem(name="sizing", subsys=SizingAssembly(), promotes=["*"])
    problem.setup()
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
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass", units="kg"
    ) == pytest.approx(22.31, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_2:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass", units="kg"
    ) == pytest.approx(96.63, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(3000.0, rel=1e-2)

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_sizing.xml"),
        problem,
    )


def test_performances_sizing_assembly_battery_enforce():

    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.pmsm.rpm"
    ] = "fastga_he.submodel.propulsion.constraints.pmsm.rpm.ensure"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.battery.state_of_charge"
    ] = "fastga_he.submodel.propulsion.constraints.battery.state_of_charge.enforce"

    ivc = get_indep_var_comp(
        list_inputs(FullSimpleAssembly(number_of_points=NB_POINTS_TEST)),
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
    model.add_subsystem(name="inputs", subsys=ivc, promotes=["*"])
    model.add_subsystem(
        name="full", subsys=FullSimpleAssembly(number_of_points=NB_POINTS_TEST), promotes=["*"]
    )

    problem.setup()
    # om.n2(problem)
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()

    write_outputs(
        pth.join(outputs.__path__[0], "full_assembly_sizing_battery_enforce.xml"),
        problem,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    ) == pytest.approx(20.28, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(2424.4, rel=1e-2)


def test_performances_sizing_assembly_battery_ensure():

    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.pmsm.rpm"
    ] = "fastga_he.submodel.propulsion.constraints.pmsm.rpm.ensure"
    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.battery.state_of_charge"
    ] = "fastga_he.submodel.propulsion.constraints.battery.state_of_charge.ensure"

    ivc = get_indep_var_comp(
        list_inputs(FullSimpleAssembly(number_of_points=NB_POINTS_TEST)),
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
    model.add_subsystem(name="inputs", subsys=ivc, promotes=["*"])
    model.add_subsystem(
        name="full", subsys=FullSimpleAssembly(number_of_points=NB_POINTS_TEST), promotes=["*"]
    )

    problem.setup()
    # om.n2(problem)
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()

    write_outputs(
        pth.join(outputs.__path__[0], "full_assembly_sizing_battery_ensure.xml"),
        problem,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min", units="percent"
    ) == pytest.approx(37.29, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(3000.0, rel=1e-2)


def test_assembly_sizing_from_pt_file():

    oad.RegisterSubmodel.active_models[
        "submodel.propulsion.constraints.pmsm.rpm"
    ] = "fastga_he.submodel.propulsion.constraints.pmsm.rpm.enforce"

    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

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
        "data:propulsion:he_power_train:DC_bus:dc_bus_2:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass", units="kg"
    ) == pytest.approx(96.63, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(3000.0, rel=1e-2)

    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        3164.28, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:CG:x", units="m") == pytest.approx(
        2.867, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:low_speed:CD0") == pytest.approx(
        0.000357, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:cruise:CD0") == pytest.approx(
        0.000352, rel=1e-2
    )

    write_outputs(
        pth.join(outputs.__path__[0], "assembly_sizing_from_pt_file.xml"),
        problem,
    )


def test_performances_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

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

    current_in = problem.get_val("component.dc_dc_converter_1.dc_current_in", units="A")
    voltage_in = problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V")

    assert current_in * voltage_in == pytest.approx(
        np.array(
            [
                187277.0,
                188236.0,
                189173.0,
                190095.0,
                191006.0,
                191907.0,
                192800.0,
                193680.0,
                194546.0,
                195395.0,
            ]
        ),
        abs=1,
    )

    write_outputs(
        pth.join(outputs.__path__[0], "assembly_performances_from_pt_file.xml"),
        problem,
    )


def test_mass_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainMassFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PowerTrainMassFromFile(power_train_file_path=pt_file_path),
        ivc,
    )

    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        5150.0, rel=1e-2
    )
    problem.check_partials(compact_print=True)


def test_cg_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainCGFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PowerTrainCGFromFile(power_train_file_path=pt_file_path),
        ivc,
    )

    assert problem.get_val("data:propulsion:he_power_train:CG:x", units="m") == pytest.approx(
        2.868, rel=1e-2
    )
    problem.check_partials(compact_print=True)


def test_drag_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainDragFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PowerTrainDragFromFile(power_train_file_path=pt_file_path),
        ivc,
    )

    assert problem.get_val("data:propulsion:he_power_train:low_speed:CD0") == pytest.approx(
        0.000357, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:cruise:CD0") == pytest.approx(
        0.000352, rel=1e-2
    )
    problem.check_partials(compact_print=True)
