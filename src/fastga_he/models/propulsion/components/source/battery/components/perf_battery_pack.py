# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..components.perf_cell_temperature import PerformancesCellTemperatureMission
from ..components.perf_direct_bus_connection import PerformancesBatteryDirectBusConnection
from ..components.perf_module_current import PerformancesModuleCurrent
from ..components.perf_open_circuit_voltage import PerformancesOpenCircuitVoltage
from ..components.perf_internal_resistance import PerformancesInternalResistance
from ..components.perf_cell_voltage import PerformancesCellVoltage
from ..components.perf_module_voltage import PerformancesModuleVoltage
from ..components.perf_battery_voltage import PerformancesBatteryVoltage
from ..components.perf_battery_c_rate import PerformancesModuleCRate
from ..components.perf_battery_relative_capacity import PerformancesRelativeCapacity
from ..components.perf_soc_decrease import PerformancesSOCDecrease
from ..components.perf_update_soc import PerformancesUpdateSOC
from ..components.perf_joule_losses import PerformancesCellJouleLosses
from ..components.perf_entropic_heat_coefficient import PerformancesEntropicHeatCoefficient
from ..components.perf_entropic_losses import PerformancesCellEntropicLosses
from ..components.perf_cell_losses import PerformancesCellLosses
from ..components.perf_battery_losses import PerformancesBatteryLosses
from ..components.perf_battery_power import PerformancesBatteryPower
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_battery_efficiency import PerformancesBatteryEfficiency
from ..components.perf_energy_consumption import PerformancesEnergyConsumption
from ..components.perf_battery_energy_consumed import PerformancesEnergyConsumed
from ..components.perf_inflight_emissions import PerformancesBatteryPackInFlightEmissions


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
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the battery is directly connected to a bus, a special mode is required to "
            "interface the two",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]
        direct_bus_connection = self.options["direct_bus_connection"]

        self.add_subsystem(
            "cell_temperature",
            PerformancesCellTemperatureMission(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )

        if self.options["direct_bus_connection"]:
            self.add_subsystem(
                "direct_bus_connection",
                PerformancesBatteryDirectBusConnection(number_of_points=number_of_points),
                promotes=["*"],
            )

        self.add_subsystem(
            "current_per_module",
            PerformancesModuleCurrent(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "battery_c_rate",
            PerformancesModuleCRate(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "battery_relative_capacity",
            PerformancesRelativeCapacity(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "battery_soc_decrease",
            PerformancesSOCDecrease(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "update_soc",
            PerformancesUpdateSOC(number_of_points=number_of_points),
            promotes=["*"],
        )
        # Though these variable depends on variables that are looped on, they don't affect the
        # value that we loop on hence why they are put here to save time.
        self.add_subsystem(
            "open_circuit_voltage",
            PerformancesOpenCircuitVoltage(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "internal_resistance",
            PerformancesInternalResistance(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "cell_voltage",
            PerformancesCellVoltage(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "module_voltage",
            PerformancesModuleVoltage(
                number_of_points=number_of_points,
                battery_pack_id=battery_pack_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "battery_voltage",
            PerformancesBatteryVoltage(
                number_of_points=number_of_points, direct_bus_connection=direct_bus_connection
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "joule_losses_cell",
            PerformancesCellJouleLosses(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "entropic_heat_coefficient",
            PerformancesEntropicHeatCoefficient(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "entropic_losses_cell",
            PerformancesCellEntropicLosses(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "losses_cell",
            PerformancesCellLosses(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "battery_losses",
            PerformancesBatteryLosses(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "battery_power",
            PerformancesBatteryPower(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(number_of_points=number_of_points, battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "efficiency",
            PerformancesBatteryEfficiency(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "energy_consumption",
            PerformancesEnergyConsumption(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "energy_consumed",
            PerformancesEnergyConsumed(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )

        fuel_consumed = om.IndepVarComp()
        fuel_consumed.add_output("fuel_consumed_t", np.full(number_of_points, 0.0), units="kg")
        self.add_subsystem(
            "fuel_consumed",
            fuel_consumed,
            promotes=["*"],
        )
        self.add_subsystem(
            name="emissions",
            subsys=PerformancesBatteryPackInFlightEmissions(
                number_of_points=number_of_points, battery_pack_id=battery_pack_id
            ),
            promotes=["*"],
        )

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        number_of_cells_module = inputs[
            "module_voltage.data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells"
        ]

        # Based on the max voltage and cut off voltage of the battery cell
        fake_cell_voltage = np.linspace(4.2, 2.65, number_of_points)
        module_voltage = fake_cell_voltage * number_of_cells_module

        outputs["module_voltage"] = module_voltage

        if self.options["direct_bus_connection"]:
            outputs["battery_voltage"] = module_voltage
        else:
            outputs["voltage_out"] = module_voltage
