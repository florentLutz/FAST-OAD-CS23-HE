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
from ..components.connectors.dc_dc_converter import PerformancesDCDCConverter

XML_FILE = "simple_assembly.xml"
NB_POINTS_TEST = 10


class PerformancesAssembly(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
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
        ivc.add_output("rpm", units="min**-1", val=np.full(number_of_points, 4000))

        ivc2 = om.IndepVarComp()
        ivc2.add_output("switching_frequency", units="Hz", val=np.full(number_of_points, 12000.0))

        ivc3 = om.IndepVarComp()
        ivc3.add_output("heat_sink_temperature", units="degK", val=np.full(NB_POINTS_TEST, 288.15))

        ivc4 = om.IndepVarComp()
        ivc4.add_output("efficiency", val=np.full(NB_POINTS_TEST, 0.98))

        ivc5 = om.IndepVarComp()
        ivc5.add_output("voltage_out_target", val=np.full(NB_POINTS_TEST, 850.0))

        ivc6 = om.IndepVarComp()
        ivc6.add_output("voltage_in", val=np.full(NB_POINTS_TEST, 860.0))

        self.add_subsystem("propeller_rot_speed", ivc, promotes=[])
        self.add_subsystem("control_inverter", ivc2, promotes=[])
        self.add_subsystem("inverter_heat_sink", ivc3, promotes=[])
        self.add_subsystem("converter_efficiency", ivc4, promotes=[])
        self.add_subsystem("converter_voltage_target", ivc5, promotes=[])
        self.add_subsystem("dc_dc_converter_voltage_in", ivc6, promotes=[])

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
                # dc_dc_converter_id="dc_dc_converter_1",
                number_of_points=number_of_points,
            ),
            promotes=[],
        )

        self.connect("propeller_rot_speed.rpm", ["propeller_1.rpm", "motor_1.rpm"])
        self.connect("control_inverter.switching_frequency", "inverter_1.switching_frequency")
        self.connect("inverter_heat_sink.heat_sink_temperature", "inverter_1.heat_sink_temperature")
        self.connect("propeller_1.shaft_power", "motor_1.shaft_power")
        self.connect("motor_1.rms_current_one_phase", "inverter_1.current")
        self.connect("motor_1.peak_voltage", "inverter_1.peak_ac_voltage")
        self.connect("motor_1.rms_voltage", "inverter_1.rms_voltage")
        self.connect("dc_bus_1.voltage", "inverter_1.dc_voltage")
        self.connect("inverter_1.dc_current", "dc_bus_1.current_out_1")
        self.connect("dc_bus_1.voltage", "dc_line_1.voltage_a")
        self.connect("dc_line_1.total_current", "dc_bus_1.current_in_1")
        self.connect("dc_bus_2.voltage", "dc_line_1.voltage_b")
        self.connect("dc_line_1.total_current", "dc_bus_2.current_out_1")
        self.connect("dc_dc_converter_1.current_out", "dc_bus_2.current_in_1")
        self.connect("dc_bus_2.voltage", "dc_dc_converter_1.voltage_out")
        self.connect("dc_dc_converter_voltage_in.voltage_in", "dc_dc_converter_1.voltage_in")
        self.connect("converter_efficiency.efficiency", "dc_dc_converter_1.efficiency")
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
        * problem.get_val("component.dc_line_1.voltage_a", units="V")
    )

    print("\n=========== DC power start of cable ===========")
    print(
        problem.get_val("component.dc_line_1.total_current", units="A")
        * problem.get_val("component.dc_line_1.voltage_b", units="V")
    )
    # Result is not really accurate since we used a ICE propeller coupled to a small PMSM not
    # sized for the demand, though it shows that the assembly works just fine

    # om.n2(problem)

    _, _, residuals = problem.model.get_nonlinear_vectors()

    assert problem.get_val("component.dc_line_1.total_current", units="A") * problem.get_val(
        "component.dc_line_1.voltage_b", units="V"
    ) == pytest.approx(
        np.array(
            [
                283233.7,
                284871.2,
                286503.9,
                288122.0,
                289725.1,
                291314.1,
                292890.0,
                294453.3,
                296004.0,
                297542.2,
            ]
        ),
        rel=1e-2,
    )
