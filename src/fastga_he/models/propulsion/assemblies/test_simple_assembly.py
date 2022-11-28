# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

import openmdao.api as om
import fastoad.api as oad
from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs
from utils.write_outputs import write_outputs

from .simple_assembly.performances_simple_assembly import PerformancesAssembly
from .simple_assembly.sizing_simple_assembly import SizingAssembly
from .simple_assembly.full_simple_assembly import FullSimpleAssembly

from . import outputs

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
                186860.0,
                187798.0,
                188721.0,
                189627.0,
                190517.0,
                191390.0,
                192246.0,
                193085.0,
                193907.0,
                194712.0,
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
    ) == pytest.approx(40.01, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass", units="kg"
    ) == pytest.approx(8.86, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_2:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass", units="kg"
    ) == pytest.approx(86.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(7936.0, rel=1e-2)

    write_outputs(
        pth.join(outputs.__path__[0], "simple_assembly_sizing.xml"),
        problem,
    )


def test_performances_sizing_assembly():

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
        pth.join(outputs.__path__[0], "full_assembly_sizing.xml"),
        problem,
    )
