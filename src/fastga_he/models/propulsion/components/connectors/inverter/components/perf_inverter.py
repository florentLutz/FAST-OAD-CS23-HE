# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_switching_frequency import PerformancesSwitchingFrequencyMission
from .perf_heat_sink_temperature import PerformancesHeatSinkTemperatureMission
from .perf_modulation_index import PerformancesModulationIndex
from .perf_resistance import PerformancesResistance
from .perf_gate_voltage import PerformancesGateVoltage
from .perf_conduction_loss import PerformancesConductionLosses
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_total_loss import PerformancesLosses
from .perf_casing_temperature import PerformancesCasingTemperature
from .perf_junction_temperature import PerformancesJunctionTemperature
from .perf_efficiency import PerformancesEfficiency
from .perf_dc_current import PerformancesDCCurrent
from .perf_maximum import PerformancesMaximum


class PerformancesInverter(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "switching_frequency",
            PerformancesSwitchingFrequencyMission(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_temperature",
            PerformancesHeatSinkTemperatureMission(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "modulation_idx",
            PerformancesModulationIndex(number_of_points=number_of_points),
            promotes=["*"],
        )
        # Switching losses do not depend on current and gate voltage so we take them out of the
        # loop to save some time
        self.add_subsystem(
            "switching_losses",
            PerformancesSwitchingLosses(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_profile",
            PerformancesInverterTemperature(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "efficiency", PerformancesEfficiency(number_of_points=number_of_points), promotes=["*"]
        )
        self.add_subsystem(
            "dc_side_current",
            PerformancesDCCurrent(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["*"],
        )


class PerformancesInverterTemperature(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.nonlinear_solver.options["stall_limit"] = 20
        self.nonlinear_solver.options["stall_tol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "resistance",
            PerformancesResistance(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "gate_voltage",
            PerformancesGateVoltage(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "conduction_losses",
            PerformancesConductionLosses(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "total_losses",
            PerformancesLosses(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_casing",
            PerformancesCasingTemperature(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_junction",
            PerformancesJunctionTemperature(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
