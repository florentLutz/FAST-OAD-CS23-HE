# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ...components.loads.pmsm import PerformancesPMSM
from ...components.propulsor.propeller import PerformancesPropeller
from ...components.connectors.inverter import PerformancesInverter
from ...components.connectors.dc_cable import PerformancesHarness
from ...components.connectors.dc_sspc import PerformancesDCSSPC
from ...components.connectors.dc_bus import PerformancesDCBus
from ...components.source.battery import PerformancesBatteryPack


class PerformancesAssemblyDirectSSPCBatteryConnection(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-8
        self.linear_solver = om.DirectSolver()

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "propeller_1",
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "thrust", "data:*", "density"],
        )
        self.add_subsystem(
            "motor_1",
            PerformancesPMSM(motor_id="motor_1", number_of_points=number_of_points),
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
            PerformancesHarness(
                harness_id="harness_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
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
            "battery_pack_1",
            PerformancesBatteryPack(
                battery_pack_id="battery_pack_1",
                number_of_points=number_of_points,
                direct_bus_connection=True,
            ),
            promotes=["data:*", "time_step"],
        )

        self.add_subsystem(
            "dc_sspc_412",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_412",
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_sspc_1",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_1", number_of_points=number_of_points, at_bus_output=False
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_sspc_2",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_2",
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_sspc_3",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_3", number_of_points=number_of_points, at_bus_output=False
            ),
            promotes=["data:*"],
        )

        self.connect("propeller_1.rpm", "motor_1.rpm")
        self.connect("propeller_1.shaft_power_in", "motor_1.shaft_power_out")

        self.connect(
            "motor_1.ac_current_rms_in_one_phase", "inverter_1.ac_current_rms_out_one_phase"
        )
        self.connect("motor_1.ac_voltage_peak_in", "inverter_1.ac_voltage_peak_out")
        self.connect("motor_1.ac_voltage_rms_in", "inverter_1.ac_voltage_rms_out")

        # INVERTER 1 to SSPC 412
        self.connect("dc_sspc_412.dc_voltage_out", "inverter_1.dc_voltage_in")
        self.connect("inverter_1.dc_current_in", "dc_sspc_412.dc_current_out")

        # SSPC 412 to DC BUS 1
        self.connect("dc_bus_1.dc_voltage", "dc_sspc_412.dc_voltage_in")
        self.connect("dc_sspc_412.dc_current_in", "dc_bus_1.dc_current_out_1")

        # DC BUS 1 TO SSPC 1
        self.connect("dc_bus_1.dc_voltage", "dc_sspc_1.dc_voltage_in")
        self.connect("dc_sspc_1.dc_current_in", "dc_bus_1.dc_current_in_1")

        # DC BUS 1 TO LINE 1
        self.connect("dc_sspc_1.dc_voltage_out", "dc_line_1.dc_voltage_out")
        self.connect("dc_line_1.dc_current", "dc_sspc_1.dc_current_out")

        self.connect("dc_sspc_2.dc_voltage_out", "dc_line_1.dc_voltage_in")
        self.connect("dc_line_1.dc_current", "dc_sspc_2.dc_current_out")

        self.connect("dc_bus_2.dc_voltage", "dc_sspc_2.dc_voltage_in")
        self.connect("dc_sspc_2.dc_current_in", "dc_bus_2.dc_current_out_1")

        # DC SSPC 3 TO BUS 2
        self.connect("dc_bus_2.dc_voltage", "dc_sspc_3.dc_voltage_in")
        self.connect("dc_sspc_3.dc_current_in", "dc_bus_2.dc_current_in_1")

        # DC SSPC 3 TO BATTERY
        self.connect("dc_sspc_3.dc_voltage_out", "battery_pack_1.voltage_out")
        self.connect("battery_pack_1.dc_current_out", "dc_sspc_3.dc_current_out")
