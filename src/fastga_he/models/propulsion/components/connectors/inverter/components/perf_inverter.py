# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

import fastoad.api as oad

from .perf_switching_frequency import PerformancesSwitchingFrequencyMission
from .perf_heat_sink_temperature import PerformancesHeatSinkTemperatureMission
from .perf_modulation_index import PerformancesModulationIndex
from .perf_resistance import PerformancesResistance
from .perf_gate_voltage import PerformancesGateVoltage
from .perf_conduction_loss import PerformancesConductionLosses
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_total_loss import PerformancesLosses
from .perf_casing_temperature import PerformancesCasingTemperature
from .perf_dc_current import PerformancesDCCurrent
from .perf_power_output import PerformancesPowerOutput
from .perf_maximum import PerformancesMaximum

from .perf_junction_temperature_fixed import SUBMODEL_INVERTER_JUNCTION_TEMPERATURE_FIXED

from ..constants import SUBMODEL_INVERTER_JUNCTION_TEMPERATURE, SUBMODEL_INVERTER_EFFICIENCY


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
        option_efficiency = {
            "inverter_id": inverter_id,
            "number_of_points": number_of_points,
        }
        self.add_subsystem(
            "efficiency",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_INVERTER_EFFICIENCY, options=option_efficiency
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_side_current",
            PerformancesDCCurrent(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "power_out",
            PerformancesPowerOutput(number_of_points=number_of_points),
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

        # Solvers setup, if temperature is fixed, there is no loop so we can omit it
        if (
            oad.RegisterSubmodel.active_models[SUBMODEL_INVERTER_JUNCTION_TEMPERATURE]
            != SUBMODEL_INVERTER_JUNCTION_TEMPERATURE_FIXED
        ):
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

        junction_temperature_option = {
            "inverter_id": inverter_id,
            "number_of_points": number_of_points,
        }

        self.add_subsystem(
            "temperature_junction",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_INVERTER_JUNCTION_TEMPERATURE, options=junction_temperature_option
            ),
            promotes=["*"],
        )
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
