# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

import fastoad.api as oad
from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs
from utils.write_outputs import write_outputs

from .simple_assembly.performances_simple_assembly_direct_bus_battery_connection import (
    PerformancesAssemblyDirectBusBatteryConnection,
)

from . import outputs

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

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

    print("\n=========== Battery SOC ===========")
    print(problem.get_val("performances.battery_pack_1.state_of_charge", units="percent"))

    print("\n=========== Battery losses ===========")
    print(problem.get_val("performances.battery_pack_1.battery_losses.losses_battery", units="W"))

    # om.n2(problem)

    _, _, residuals = problem.model.performances.get_nonlinear_vectors()

    assert problem.get_val(
        "performances.battery_pack_1.dc_current_out", units="A"
    ) * problem.get_val("performances.battery_pack_1.voltage_out", units="V") == pytest.approx(
        np.array(
            [
                200984.68111555,
                201982.65435534,
                202963.33255123,
                203927.24227857,
                204874.79738294,
                205806.24725618,
                206721.66032759,
                207620.91851917,
                208503.71739528,
                209369.58886383,
            ]
        ),
        abs=1,
    )

    write_outputs(
        pth.join(
            outputs.__path__[0], "simple_assembly_direct_bus_battery_connection_performances.xml"
        ),
        problem,
    )
