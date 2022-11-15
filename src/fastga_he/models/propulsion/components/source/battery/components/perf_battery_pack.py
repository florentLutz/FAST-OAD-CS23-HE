# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_module_current import PerformancesModuleCurrent
from ..components.perf_open_circuit_voltage import PerformancesOpenCircuitVoltage
from ..components.perf_internal_resistance import PerformancesInternalResistance
from ..components.perf_cell_voltage import PerformancesCellVoltage
from ..components.perf_module_voltage import PerformancesModuleVoltage
from ..components.perf_battery_voltage import PerformancesBatteryVoltage


class PerformancesBatteryPack(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_subsystem(
            "current_per_module",
            PerformancesModuleCurrent(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["data:*", "current_out"],
        )
        self.add_subsystem(
            "open_circuit_voltage",
            PerformancesOpenCircuitVoltage(number_of_points=number_of_points),
            promotes=["state_of_charge"],
        )
        self.add_subsystem(
            "internal_resistance",
            PerformancesInternalResistance(number_of_points=number_of_points),
            promotes=["state_of_charge"],
        )
        self.add_subsystem(
            "cell_voltage",
            PerformancesCellVoltage(number_of_points=number_of_points),
            promotes=[],
        )
        self.add_subsystem(
            "module_voltage",
            PerformancesModuleVoltage(
                number_of_points=number_of_points,
                battery_pack_id=battery_pack_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "battery_voltage",
            PerformancesBatteryVoltage(number_of_points=number_of_points),
            promotes=["voltage_out"],
        )

        self.connect("current_per_module.current_one_module", "cell_voltage.current_one_module")
        self.connect(
            "open_circuit_voltage.open_circuit_voltage", "cell_voltage.open_circuit_voltage"
        )
        self.connect("internal_resistance.internal_resistance", "cell_voltage.internal_resistance")
        self.connect("cell_voltage.terminal_voltage", "module_voltage.terminal_voltage")
        self.connect("module_voltage.module_voltage", "battery_voltage.module_voltage")
