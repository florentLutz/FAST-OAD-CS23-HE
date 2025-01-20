# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ...components.loads.pmsm import PerformancesPMSM
from ...components.propulsor.propeller import PerformancesPropeller
from ...components.connectors.inverter import PerformancesInverter
from ...components.connectors.dc_cable import PerformancesHarness
from ...components.tanks.hydrogen_gas_tank import PerformancesHydrogenGasTank
from ...components.connectors.dc_bus import PerformancesDCBus
from ...components.connectors.dc_dc_converter import PerformancesDCDCConverter
from ...components.connectors.dc_sspc import PerformancesDCSSPC
from ...components.source.pemfc import PerformancesPEMFCStack


class PerformancesAssembly(om.Group):
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
            "dc_sspc_412",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_412",
                number_of_points=number_of_points,
            ),
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
            "dc_sspc_1",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_1", number_of_points=number_of_points, at_bus_output=False
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
            "dc_sspc_2",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_2",
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
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
            "dc_sspc_1337",
            PerformancesDCSSPC(
                dc_sspc_id="dc_sspc_1337", number_of_points=number_of_points, at_bus_output=False
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
            "pemfc_stack_1",
            PerformancesPEMFCStack(
                pemfc_stack_id="pemfc_stack_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step"],
        )
        self.add_subsystem(
            "hydrogen_gas_tank_1",
            PerformancesHydrogenGasTank(
                hydrogen_gas_tank_id="hydrogen_gas_tank_1",
                number_of_points=number_of_points,
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

        # INVERTER 1 to DC BUS 1
        # self.connect("dc_bus_1.dc_voltage", "inverter_1.dc_voltage_in")
        # self.connect("inverter_1.dc_current_in", "dc_bus_1.dc_current_out_1")

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

        # DC BUS 2 TO DC DC CONVERTER
        # self.connect("dc_dc_converter_1.dc_current_out", "dc_bus_2.dc_current_in_1")
        # self.connect("dc_bus_2.dc_voltage", "dc_dc_converter_1.dc_voltage_out")

        # DC BUS 2 TO SSPC 1337
        self.connect("dc_bus_2.dc_voltage", "dc_sspc_1337.dc_voltage_in")
        self.connect("dc_sspc_1337.dc_current_in", "dc_bus_2.dc_current_in_1")

        # SSPC 1337 TO DC DC CONVERTER
        self.connect("dc_sspc_1337.dc_voltage_out", "dc_dc_converter_1.dc_voltage_out")
        self.connect("dc_dc_converter_1.dc_current_out", "dc_sspc_1337.dc_current_out")

        self.connect("pemfc_stack_1.voltage_out", "dc_dc_converter_1.dc_voltage_in")
        self.connect("dc_dc_converter_1.dc_current_in", "pemfc_stack_1.dc_current_out")

        # PEMFC TO HYDROGEN GAS TANK
        self.connect("pemfc_stack_1.fuel_consumed_t", "hydrogen_gas_tank_1.fuel_consumed_t")
