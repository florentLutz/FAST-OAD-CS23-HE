# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import pytest
from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..components.loads.pmsm import PerformancePMSM, SizingPMSM
from ..components.propulsor.propeller import PerformancesPropeller, SizingPropeller
from ..components.connectors.inverter import PerformancesInverter, SizingInverter
from ..components.connectors.dc_cable import PerformanceHarness, SizingHarness
from ..components.connectors.dc_bus import PerformancesDCBus, SizingDCBus
from ..components.connectors.dc_dc_converter import PerformancesDCDCConverter, SizingDCDCConverter
from ..components.source.battery import PerformancesBatteryPack, SizingBatteryPack

XML_FILE = "simple_assembly.xml"
NB_POINTS_TEST = 10


class PerformancesAssembly(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        ivc = om.IndepVarComp()
        ivc.add_output("rpm", units="min**-1", val=np.full(number_of_points, 2000))

        ivc2 = om.IndepVarComp()
        ivc2.add_output("switching_frequency", units="Hz", val=np.full(number_of_points, 12000.0))

        ivc3 = om.IndepVarComp()
        ivc3.add_output("heat_sink_temperature", units="degK", val=np.full(NB_POINTS_TEST, 288.15))

        ivc4 = om.IndepVarComp()
        ivc4.add_output("switching_frequency", units="Hz", val=np.full(NB_POINTS_TEST, 12000))

        ivc5 = om.IndepVarComp()
        ivc5.add_output("voltage_out_target", val=np.full(NB_POINTS_TEST, 850.0))

        ivc6 = om.IndepVarComp()
        ivc6.add_output("cell_temperature", val=np.full(NB_POINTS_TEST, 288.15), units="degK")

        self.add_subsystem("propeller_rot_speed", ivc, promotes=[])
        self.add_subsystem("control_inverter", ivc2, promotes=[])
        self.add_subsystem("inverter_heat_sink", ivc3, promotes=[])
        self.add_subsystem("control_converter", ivc4, promotes=[])
        self.add_subsystem("converter_voltage_target", ivc5, promotes=[])
        self.add_subsystem("battery_temperature", ivc6, promotes=[])

        self.add_subsystem(
            "propeller_1",
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "thrust", "data:*"],
        )
        self.add_subsystem(
            "motor_1",
            PerformancePMSM(motor_id="motor_1", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_1",
            PerformancesInverter(inverter_id="inverter_1", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_bus_1",
            PerformancesDCBus(
                dc_bus_id="dc_bus_1",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=1,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_line_1",
            PerformanceHarness(
                harness_id="harness_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_bus_2",
            PerformancesDCBus(
                dc_bus_id="dc_bus_2",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=1,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_dc_converter_1",
            PerformancesDCDCConverter(
                dc_dc_converter_id="dc_dc_converter_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "battery_pack_1",
            PerformancesBatteryPack(
                battery_pack_id="battery_pack_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step"],
        )

        self.connect("propeller_rot_speed.rpm", ["propeller_1.rpm", "motor_1.rpm"])
        self.connect("control_inverter.switching_frequency", "inverter_1.switching_frequency")
        self.connect("inverter_heat_sink.heat_sink_temperature", "inverter_1.heat_sink_temperature")
        self.connect(
            "control_converter.switching_frequency", "dc_dc_converter_1.switching_frequency"
        )
        self.connect("propeller_1.shaft_power", "motor_1.shaft_power")
        self.connect("motor_1.rms_current_one_phase", "inverter_1.current")
        self.connect("motor_1.peak_voltage", "inverter_1.peak_ac_voltage")
        self.connect("motor_1.rms_voltage", "inverter_1.rms_voltage")
        self.connect("dc_bus_1.voltage", "inverter_1.dc_voltage")
        self.connect("inverter_1.dc_current", "dc_bus_1.current_out_1")
        self.connect("dc_bus_1.voltage", "dc_line_1.voltage_out")
        self.connect("dc_line_1.total_current", "dc_bus_1.current_in_1")
        self.connect("dc_bus_2.voltage", "dc_line_1.voltage_in")
        self.connect("dc_line_1.total_current", "dc_bus_2.current_out_1")
        self.connect("dc_dc_converter_1.current_out", "dc_bus_2.current_in_1")
        self.connect("dc_bus_2.voltage", "dc_dc_converter_1.voltage_out")
        self.connect(
            "converter_voltage_target.voltage_out_target", "dc_dc_converter_1.voltage_out_target"
        )
        self.connect("battery_pack_1.voltage_out", "dc_dc_converter_1.voltage_in")
        self.connect("dc_dc_converter_1.current_in", "battery_pack_1.current_out")
        self.connect("battery_temperature.cell_temperature", "battery_pack_1.cell_temperature")


def test_assembly():

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

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAssembly(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    print("\n=========== Propulsive power ===========")
    print(problem.get_val("true_airspeed", units="m/s") * problem.get_val("thrust", units="N"))

    print("\n=========== Shaft power ===========")
    print(problem.get_val("component.propeller_1.shaft_power", units="W"))

    print("\n=========== AC power ===========")
    print(
        problem.get_val("component.motor_1.rms_current", units="A")
        * problem.get_val("component.motor_1.rms_voltage", units="V")
    )

    print("\n=========== DC power before inverter ===========")
    print(
        problem.get_val("component.inverter_1.dc_current", units="A")
        * problem.get_val("component.inverter_1.dc_voltage", units="V")
    )

    print("\n=========== DC power before bus/end of cable ===========")
    print(
        problem.get_val("component.dc_line_1.total_current", units="A")
        * problem.get_val("component.dc_line_1.voltage_out", units="V")
    )

    print("\n=========== DC power start of cable ===========")
    print(
        problem.get_val("component.dc_line_1.total_current", units="A")
        * problem.get_val("component.dc_line_1.voltage_in", units="V")
    )
    # Result is not really accurate since we used a ICE propeller coupled to a small PMSM not
    # sized for the demand, though it shows that the assembly works just fine

    print("\n=========== DC power input of the DC/DC converter ===========")
    print(
        problem.get_val("component.dc_dc_converter_1.current_in", units="A")
        * problem.get_val("component.dc_dc_converter_1.voltage_in", units="V")
    )

    print("\n=========== Battery SOC ===========")
    print(problem.get_val("component.battery_pack_1.state_of_charge", units="percent"))

    print("\n=========== Battery losses ===========")
    print(problem.get_val("component.battery_pack_1.battery_losses.losses_battery", units="W"))

    # om.n2(problem)

    _, _, residuals = problem.model.component.get_nonlinear_vectors()

    assert problem.get_val("component.dc_dc_converter_1.current_in", units="A") * problem.get_val(
        "component.dc_dc_converter_1.voltage_in", units="V"
    ) == pytest.approx(
        np.array(
            [
                186568.9,
                187505.3,
                188425.1,
                189328.3,
                190214.8,
                191084.6,
                191937.5,
                192773.5,
                193592.5,
                194394.5,
            ]
        ),
        abs=1,
    )
