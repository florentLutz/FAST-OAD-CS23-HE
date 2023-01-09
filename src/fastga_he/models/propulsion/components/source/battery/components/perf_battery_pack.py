# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
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
from ..components.perf_energy_consumption import PerformancesEnergyConsumption


class PerformancesBatteryPack(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.LinearBlockGS()

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
            "battery_c_rate",
            PerformancesModuleCRate(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["data:*", "c_rate"],
        )
        self.add_subsystem(
            "battery_soc_decrease",
            PerformancesSOCDecrease(number_of_points=number_of_points),
            promotes=["time_step", "c_rate"],
        )
        self.add_subsystem(
            "update_soc",
            PerformancesUpdateSOC(number_of_points=number_of_points),
            promotes=["state_of_charge"],
        )
        # Though these variable depends on variables that are looped on, they don't affect the
        # value that we loop on hence why they are put here to save time.
        self.add_subsystem(
            "open_circuit_voltage",
            PerformancesOpenCircuitVoltage(number_of_points=number_of_points),
            promotes=["state_of_charge", "open_circuit_voltage"],
        )
        self.add_subsystem(
            "internal_resistance",
            PerformancesInternalResistance(number_of_points=number_of_points),
            promotes=["state_of_charge", "internal_resistance"],
        )
        self.add_subsystem(
            "cell_voltage",
            PerformancesCellVoltage(number_of_points=number_of_points),
            promotes=["internal_resistance", "open_circuit_voltage"],
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
            "joule_losses_cell",
            PerformancesCellJouleLosses(number_of_points=number_of_points),
            promotes=["internal_resistance"],
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
            promotes=["data:*", "state_of_charge", "c_rate"],
        )
        self.add_subsystem(
            "energy_consumption",
            PerformancesEnergyConsumption(number_of_points=number_of_points),
            promotes=["dc_current_out", "voltage_out", "time_step", "non_consumable_energy_t"],
        )

        fuel_consumed = om.IndepVarComp()
        fuel_consumed.add_output(
            "fuel_consumed_t",
            np.full(number_of_points, 0.0),
        )
        self.add_subsystem(
            "fuel_consumed",
            fuel_consumed,
            promotes=["fuel_consumed_t"],
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
            "cell_voltage.terminal_voltage",
            ["module_voltage.terminal_voltage", "maximum.terminal_voltage"],
        )
        self.connect("module_voltage.module_voltage", "battery_voltage.module_voltage")

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
