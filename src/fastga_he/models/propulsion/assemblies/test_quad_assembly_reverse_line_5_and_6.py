# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..components.loads.pmsm import PerformancesPMSM, SizingPMSM
from ..components.propulsor.propeller import PerformancesPropeller, SizingPropeller
from ..components.connectors.inverter import PerformancesInverter, SizingInverter
from ..components.connectors.dc_cable import PerformancesHarness, SizingHarness
from ..components.connectors.dc_bus import PerformancesDCBus, SizingDCBus
from ..components.connectors.dc_dc_converter import PerformancesDCDCConverter, SizingDCDCConverter

XML_FILE = "quad_assembly.xml"
NB_POINTS_TEST = 25
COEFF_DIFF = 0.0


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
        ivc6.add_output("voltage_in", val=np.full(NB_POINTS_TEST, 860.0))

        self.add_subsystem("propeller_rot_speed", ivc, promotes=[])
        self.add_subsystem("control_inverter", ivc2, promotes=[])
        self.add_subsystem("inverter_heat_sink", ivc3, promotes=[])
        self.add_subsystem("control_converter", ivc4, promotes=[])
        self.add_subsystem("converter_voltage_target", ivc5, promotes=[])
        self.add_subsystem("dc_dc_converter_voltage_in", ivc6, promotes=[])

        ivc_thrust_1 = om.IndepVarComp(
            "thrust_1",
            val=np.linspace(
                500 * (1 - 2.0 * COEFF_DIFF), 550 * (1 - 2.0 * COEFF_DIFF), NB_POINTS_TEST
            ),
        )
        ivc_thrust_2 = om.IndepVarComp(
            "thrust_2",
            val=np.linspace(500 * (1 - COEFF_DIFF), 550 * (1 - COEFF_DIFF), NB_POINTS_TEST),
        )
        ivc_thrust_3 = om.IndepVarComp(
            "thrust_3",
            val=np.linspace(500 * (1 + COEFF_DIFF), 550 * (1 + COEFF_DIFF), NB_POINTS_TEST),
        )
        ivc_thrust_4 = om.IndepVarComp(
            "thrust_4",
            val=np.linspace(
                500 * (1 + 2.0 * COEFF_DIFF), 550 * (1 + 2.0 * COEFF_DIFF), NB_POINTS_TEST
            ),
        )

        self.add_subsystem("ivc_thrust_1", ivc_thrust_1, promotes=[])
        self.add_subsystem("ivc_thrust_2", ivc_thrust_2, promotes=[])
        self.add_subsystem("ivc_thrust_3", ivc_thrust_3, promotes=[])
        self.add_subsystem("ivc_thrust_4", ivc_thrust_4, promotes=[])

        # Propellers
        self.add_subsystem(
            "propeller_1",
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*"],
        )
        self.add_subsystem(
            "propeller_2",
            PerformancesPropeller(propeller_id="propeller_2", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*"],
        )
        self.add_subsystem(
            "propeller_3",
            PerformancesPropeller(propeller_id="propeller_3", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*"],
        )
        self.add_subsystem(
            "propeller_4",
            PerformancesPropeller(propeller_id="propeller_4", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*"],
        )

        # Motors
        self.add_subsystem(
            "motor_1",
            PerformancesPMSM(motor_id="motor_1", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "motor_2",
            PerformancesPMSM(motor_id="motor_2", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "motor_3",
            PerformancesPMSM(motor_id="motor_3", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "motor_4",
            PerformancesPMSM(motor_id="motor_4", number_of_points=number_of_points),
            promotes=["data:*"],
        )

        # Inverters
        self.add_subsystem(
            "inverter_1",
            PerformancesInverter(inverter_id="inverter_1", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_2",
            PerformancesInverter(inverter_id="inverter_2", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_3",
            PerformancesInverter(inverter_id="inverter_3", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_4",
            PerformancesInverter(inverter_id="inverter_4", number_of_points=number_of_points),
            promotes=["data:*"],
        )

        # DC Buses
        self.add_subsystem(
            "dc_bus_1",
            PerformancesDCBus(
                dc_bus_id="dc_bus_1",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=2,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_bus_2",
            PerformancesDCBus(
                dc_bus_id="dc_bus_2",
                number_of_points=number_of_points,
                number_of_inputs=2,
                number_of_outputs=1,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_bus_3",
            PerformancesDCBus(
                dc_bus_id="dc_bus_3",
                number_of_points=number_of_points,
                number_of_inputs=2,
                number_of_outputs=1,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_bus_4",
            PerformancesDCBus(
                dc_bus_id="dc_bus_4",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=2,
            ),
            promotes=["data:*"],
        )

        # DC lines
        self.add_subsystem(
            "dc_line_1",
            PerformancesHarness(
                harness_id="harness_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_2",
            PerformancesHarness(
                harness_id="harness_2",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_3",
            PerformancesHarness(
                harness_id="harness_3",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_4",
            PerformancesHarness(
                harness_id="harness_4",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_5",
            PerformancesHarness(
                harness_id="harness_5",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_6",
            PerformancesHarness(
                harness_id="harness_6",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step", "exterior_temperature"],
        )

        # Source bus
        self.add_subsystem(
            "dc_bus_5",
            PerformancesDCBus(
                dc_bus_id="dc_bus_5",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=4,
            ),
            promotes=["data:*"],
        )

        # Source converter
        self.add_subsystem(
            "dc_dc_converter_1",
            PerformancesDCDCConverter(
                dc_dc_converter_id="dc_dc_converter_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )

        self.connect(
            "propeller_rot_speed.rpm",
            [
                "propeller_1.rpm",
                "motor_1.rpm",
                "propeller_2.rpm",
                "motor_2.rpm",
                "propeller_3.rpm",
                "motor_3.rpm",
                "propeller_4.rpm",
                "motor_4.rpm",
            ],
        )

        self.connect(
            "control_inverter.switching_frequency",
            [
                "inverter_1.switching_frequency",
                "inverter_2.switching_frequency",
                "inverter_3.switching_frequency",
                "inverter_4.switching_frequency",
            ],
        )

        self.connect(
            "inverter_heat_sink.heat_sink_temperature",
            [
                "inverter_1.heat_sink_temperature",
                "inverter_2.heat_sink_temperature",
                "inverter_3.heat_sink_temperature",
                "inverter_4.heat_sink_temperature",
            ],
        )

        self.connect("ivc_thrust_1.thrust_1", "propeller_1.thrust")
        self.connect("ivc_thrust_2.thrust_2", "propeller_2.thrust")
        self.connect("ivc_thrust_3.thrust_3", "propeller_3.thrust")
        self.connect("ivc_thrust_4.thrust_4", "propeller_4.thrust")

        self.connect("propeller_1.shaft_power_in", "motor_1.shaft_power_out")
        self.connect("propeller_2.shaft_power_in", "motor_2.shaft_power_out")
        self.connect("propeller_3.shaft_power_in", "motor_3.shaft_power_out")
        self.connect("propeller_4.shaft_power_in", "motor_4.shaft_power_out")

        self.connect(
            "motor_1.ac_current_rms_in_one_phase", "inverter_1.ac_current_rms_out_one_phase"
        )
        self.connect(
            "motor_2.ac_current_rms_in_one_phase", "inverter_2.ac_current_rms_out_one_phase"
        )
        self.connect(
            "motor_3.ac_current_rms_in_one_phase", "inverter_3.ac_current_rms_out_one_phase"
        )
        self.connect(
            "motor_4.ac_current_rms_in_one_phase", "inverter_4.ac_current_rms_out_one_phase"
        )

        self.connect("motor_1.ac_voltage_peak_in", "inverter_1.ac_voltage_peak_out")
        self.connect("motor_2.ac_voltage_peak_in", "inverter_2.ac_voltage_peak_out")
        self.connect("motor_3.ac_voltage_peak_in", "inverter_3.ac_voltage_peak_out")
        self.connect("motor_4.ac_voltage_peak_in", "inverter_4.ac_voltage_peak_out")

        self.connect("motor_1.ac_voltage_rms_in", "inverter_1.ac_voltage_rms_out")
        self.connect("motor_2.ac_voltage_rms_in", "inverter_2.ac_voltage_rms_out")
        self.connect("motor_3.ac_voltage_rms_in", "inverter_3.ac_voltage_rms_out")
        self.connect("motor_4.ac_voltage_rms_in", "inverter_4.ac_voltage_rms_out")

        self.connect("dc_bus_1.dc_voltage", "inverter_1.dc_voltage_in")
        self.connect("dc_bus_2.dc_voltage", "inverter_2.dc_voltage_in")
        self.connect("dc_bus_3.dc_voltage", "inverter_3.dc_voltage_in")
        self.connect("dc_bus_4.dc_voltage", "inverter_4.dc_voltage_in")

        self.connect("inverter_1.dc_current_in", "dc_bus_1.dc_current_out_1")
        self.connect("inverter_2.dc_current_in", "dc_bus_2.dc_current_out_1")
        self.connect("inverter_3.dc_current_in", "dc_bus_3.dc_current_out_1")
        self.connect("inverter_4.dc_current_in", "dc_bus_4.dc_current_out_1")

        # DC bus 1
        self.connect("dc_bus_1.dc_voltage", "dc_line_1.dc_voltage_out")
        self.connect("dc_bus_1.dc_voltage", "dc_line_5.dc_voltage_in")
        self.connect("dc_line_1.dc_current", "dc_bus_1.dc_current_in_1")
        self.connect("dc_line_5.dc_current", "dc_bus_1.dc_current_out_2")

        # DC bus 2
        self.connect("dc_bus_2.dc_voltage", "dc_line_2.dc_voltage_out")
        self.connect("dc_bus_2.dc_voltage", "dc_line_5.dc_voltage_out")
        self.connect("dc_line_2.dc_current", "dc_bus_2.dc_current_in_1")
        self.connect("dc_line_5.dc_current", "dc_bus_2.dc_current_in_2")

        # DC bus 3
        self.connect("dc_bus_3.dc_voltage", "dc_line_3.dc_voltage_out")
        self.connect("dc_bus_3.dc_voltage", "dc_line_6.dc_voltage_out")
        self.connect("dc_line_3.dc_current", "dc_bus_3.dc_current_in_1")
        self.connect("dc_line_6.dc_current", "dc_bus_3.dc_current_in_2")

        # DC bus 4
        self.connect("dc_bus_4.dc_voltage", "dc_line_4.dc_voltage_out")
        self.connect("dc_bus_4.dc_voltage", "dc_line_6.dc_voltage_in")
        self.connect("dc_line_4.dc_current", "dc_bus_4.dc_current_in_1")
        self.connect("dc_line_6.dc_current", "dc_bus_4.dc_current_out_2")

        self.connect("dc_bus_5.dc_voltage", "dc_line_1.dc_voltage_in")
        self.connect("dc_bus_5.dc_voltage", "dc_line_2.dc_voltage_in")
        self.connect("dc_bus_5.dc_voltage", "dc_line_3.dc_voltage_in")
        self.connect("dc_bus_5.dc_voltage", "dc_line_4.dc_voltage_in")

        self.connect("dc_line_1.dc_current", "dc_bus_5.dc_current_out_1")
        self.connect("dc_line_2.dc_current", "dc_bus_5.dc_current_out_2")
        self.connect("dc_line_3.dc_current", "dc_bus_5.dc_current_out_3")
        self.connect("dc_line_4.dc_current", "dc_bus_5.dc_current_out_4")

        self.connect("dc_dc_converter_1.dc_current_out", "dc_bus_5.dc_current_in_1")
        self.connect("dc_bus_5.dc_voltage", "dc_dc_converter_1.dc_voltage_out")

        self.connect("dc_dc_converter_voltage_in.voltage_in", "dc_dc_converter_1.dc_voltage_in")
        self.connect(
            "control_converter.switching_frequency", "dc_dc_converter_1.switching_frequency"
        )
        self.connect(
            "converter_voltage_target.voltage_out_target", "dc_dc_converter_1.voltage_out_target"
        )


def test_assembly():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssembly(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")

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
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_1.thrust", units="N")[0]
    )
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_2.thrust", units="N")[0]
    )
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_3.thrust", units="N")[0]
    )
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_4.thrust", units="N")[0]
    )

    print("\n=========== Shaft power ===========")
    print(problem.get_val("component.propeller_1.shaft_power_in", units="W")[0])
    print(problem.get_val("component.propeller_2.shaft_power_in", units="W")[0])
    print(problem.get_val("component.propeller_3.shaft_power_in", units="W")[0])
    print(problem.get_val("component.propeller_4.shaft_power_in", units="W")[0])

    print("\n=========== AC power ===========")
    print(
        problem.get_val("component.motor_1.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_1.ac_voltage_rms_in", units="V")[0]
    )
    print(
        problem.get_val("component.motor_2.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_2.ac_voltage_rms_in", units="V")[0]
    )
    print(
        problem.get_val("component.motor_3.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_3.ac_voltage_rms_in", units="V")[0]
    )
    print(
        problem.get_val("component.motor_4.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_4.ac_voltage_rms_in", units="V")[0]
    )

    print("\n=========== DC power before inverter ===========")
    print(problem.get_val("component.inverter_1.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_1.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_1.dc_voltage_in", units="V")[0]
    )
    print(problem.get_val("component.inverter_2.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_2.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_2.dc_voltage_in", units="V")[0]
    )
    print(problem.get_val("component.inverter_3.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_3.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_3.dc_voltage_in", units="V")[0]
    )
    print(problem.get_val("component.inverter_4.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_4.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_4.dc_voltage_in", units="V")[0]
    )

    print("\n=========== DC currents in cables ===========")
    print(problem.get_val("component.dc_line_1.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_2.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_3.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_4.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_5.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_6.dc_current", units="A")[0])

    print("\n=========== DC power before bus/end of cable ===========")
    print(
        problem.get_val("component.dc_line_1.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_1.dc_voltage_out", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_2.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_2.dc_voltage_out", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_3.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_3.dc_voltage_out", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_4.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_4.dc_voltage_out", units="V")[0]
    )

    print("\n=========== DC power start of cable ===========")
    print(
        problem.get_val("component.dc_line_1.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_1.dc_voltage_in", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_2.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_2.dc_voltage_in", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_3.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_3.dc_voltage_in", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_4.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_4.dc_voltage_in", units="V")[0]
    )

    print("\n=========== DC power after converter ===========")
    print(
        problem.get_val("component.dc_dc_converter_1.dc_current_out", units="A")[0]
        * problem.get_val("component.dc_dc_converter_1.dc_voltage_out", units="V")[0]
    )

    print("\n=========== DC power before converter ===========")
    print(
        problem.get_val("component.dc_dc_converter_1.dc_current_in", units="A")[0]
        * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V")[0]
    )
    # Result is not really accurate since we used a ICE propeller coupled to a small PMSM not
    # sized for the demand, though it shows that the assembly works just fine

    # om.n2(problem)
