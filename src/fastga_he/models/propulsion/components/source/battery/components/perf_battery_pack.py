# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_cell_temperature import PerformancesCellTemperatureMission
from ..components.perf_module_current import PerformancesModuleCurrent
from ..components.perf_open_circuit_voltage import PerformancesOpenCircuitVoltage
from ..components.perf_internal_resistance import PerformancesInternalResistance
from ..components.perf_cell_voltage import PerformancesCellVoltage
from ..components.perf_module_voltage import PerformancesModuleVoltage
from ..components.perf_battery_voltage import PerformancesBatteryVoltage
from ..components.perf_battery_c_rate import PerformancesModuleCRate
from ..components.perf_soc_decrease import PerformancesSOCDecrease
from ..components.perf_update_soc import PerformancesUpdateSOC
from ..components.perf_joule_losses import PerformancesCellJouleLosses
from ..components.perf_entropic_heat_coefficient import PerformancesEntropicHeatCoefficient
from ..components.perf_entropic_losses import PerformancesCellEntropicLosses
from ..components.perf_cell_losses import PerformancesCellLosses
from ..components.perf_battery_losses import PerformancesBatteryLosses
from ..components.perf_maximum import PerformancesMaximum


class PerformancesBatteryPack(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.BroydenSolver()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()

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
            "cell_temperature",
            PerformancesCellTemperatureMission(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "current_per_module",
            PerformancesModuleCurrent(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["data:*", "dc_current_out"],
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

        self.add_subsystem(
            "battery_c_rate",
            PerformancesModuleCRate(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "battery_soc_decrease",
            PerformancesSOCDecrease(number_of_points=number_of_points),
            promotes=["time_step"],
        )
        self.add_subsystem(
            "update_soc",
            PerformancesUpdateSOC(number_of_points=number_of_points),
            promotes=["state_of_charge"],
        )
        self.add_subsystem(
            "joule_losses_cell",
            PerformancesCellJouleLosses(number_of_points=number_of_points),
            promotes=[],
        )
        self.add_subsystem(
            "entropic_heat_coefficient",
            PerformancesEntropicHeatCoefficient(number_of_points=number_of_points),
            promotes=["state_of_charge"],
        )
        self.add_subsystem(
            "entropic_losses_cell",
            PerformancesCellEntropicLosses(number_of_points=number_of_points),
            promotes=["cell_temperature"],
        )
        self.add_subsystem(
            "losses_cell",
            PerformancesCellLosses(number_of_points=number_of_points),
            promotes=[],
        )
        self.add_subsystem(
            "battery_losses",
            PerformancesBatteryLosses(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(number_of_points=number_of_points, battery_pack_id=battery_pack_id),
            promotes=["data:*", "state_of_charge"],
        )

        self.connect(
            "current_per_module.current_one_module",
            [
                "cell_voltage.current_one_module",
                "battery_c_rate.current_one_module",
                "joule_losses_cell.current_one_module",
                "entropic_losses_cell.current_one_module",
            ],
        )
        self.connect(
            "open_circuit_voltage.open_circuit_voltage", "cell_voltage.open_circuit_voltage"
        )
        self.connect(
            "internal_resistance.internal_resistance",
            ["cell_voltage.internal_resistance", "joule_losses_cell.internal_resistance"],
        )
        self.connect(
            "cell_voltage.terminal_voltage",
            ["module_voltage.terminal_voltage", "maximum.terminal_voltage"],
        )
        self.connect("module_voltage.module_voltage", "battery_voltage.module_voltage")

        self.connect("battery_c_rate.c_rate", ["battery_soc_decrease.c_rate", "maximum.c_rate"])
        self.connect(
            "battery_soc_decrease.state_of_charge_decrease", "update_soc.state_of_charge_decrease"
        )

        self.connect(
            "entropic_heat_coefficient.entropic_heat_coefficient",
            "entropic_losses_cell.entropic_heat_coefficient",
        )
        self.connect(
            "joule_losses_cell.joule_losses_cell",
            "losses_cell.joule_losses_cell",
        )
        self.connect(
            "entropic_losses_cell.entropic_losses_cell",
            "losses_cell.entropic_losses_cell",
        )
        self.connect(
            "losses_cell.losses_cell",
            ["battery_losses.losses_cell", "maximum.losses_cell"],
        )
