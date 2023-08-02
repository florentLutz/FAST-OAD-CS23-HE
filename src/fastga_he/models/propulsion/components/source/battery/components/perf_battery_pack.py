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
from ..components.perf_battery_relative_capacity import PerformancesRelativeCapacity
from ..components.perf_soc_decrease import PerformancesSOCDecrease
from ..components.perf_update_soc import PerformancesUpdateSOC
from ..components.perf_joule_losses import PerformancesCellJouleLosses
from ..components.perf_entropic_heat_coefficient import PerformancesEntropicHeatCoefficient
from ..components.perf_entropic_losses import PerformancesCellEntropicLosses
from ..components.perf_cell_losses import PerformancesCellLosses
from ..components.perf_battery_losses import PerformancesBatteryLosses
from ..components.perf_maximum import PerformancesMaximum
from ..components.perf_battery_efficiency import PerformancesBatteryEfficiency
from ..components.perf_energy_consumption import PerformancesEnergyConsumption


class PerformancesBatteryPack(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.nonlinear_solver.options["atol"] = 1e-5
        self.nonlinear_solver.options["stall_limit"] = 20
        self.nonlinear_solver.options["stall_tol"] = 1e-5
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
            PerformancesBatteryVoltage(number_of_points=number_of_points),
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

        fuel_consumed = om.IndepVarComp()
        fuel_consumed.add_output("fuel_consumed_t", np.full(number_of_points, 0.0), units="kg")
        self.add_subsystem(
            "fuel_consumed",
            fuel_consumed,
            promotes=["*"],
        )
