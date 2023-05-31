# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_voltage_out_target import PerformancesVoltageOutTargetMission
from .perf_switching_frequency import PerformancesSwitchingFrequencyMission
from .perf_heat_sink_temperature import PerformancesHeatSinkTemperatureMission
from .perf_modulation_index import PerformancesModulationIndex
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_resistance import PerformancesResistance
from .perf_gate_voltage import PerformancesGateVoltage
from .perf_conduction_loss import PerformancesConductionLosses
from .perf_total_loss import PerformancesLosses
from .perf_casing_temperature import PerformancesCasingTemperature
from .perf_junction_temperature import PerformancesJunctionTemperature
from .perf_efficiency import PerformancesEfficiency
from .perf_load_side import PerformancesRectifierLoadSide
from .perf_generator_side import PerformancesRectifierGeneratorSide
from .perf_rectifier_relations import PerformancesRectifierRelations
from .perf_maximum import PerformancesMaximum


class PerformancesRectifier(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "voltage_out_target",
            PerformancesVoltageOutTargetMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "switching_frequency",
            PerformancesSwitchingFrequencyMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_temperature",
            PerformancesHeatSinkTemperatureMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "modulation_idx",
            PerformancesModulationIndex(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "switching_losses",
            PerformancesSwitchingLosses(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_profile",
            PerformancesRectifierTemperature(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "efficiency",
            PerformancesEfficiency(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "load_side",
            PerformancesRectifierLoadSide(number_of_points=number_of_points),
            promotes=["ac_voltage_rms_in", "ac_current_rms_in_one_phase"],
        )
        self.add_subsystem(
            "generator_side",
            PerformancesRectifierGeneratorSide(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "dc_current_out"],
        )
        self.add_subsystem(
            "converter_relation",
            PerformancesRectifierRelations(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "voltage_out_target", "dc_current_out", "efficiency"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(number_of_points=number_of_points, rectifier_id=rectifier_id),
            promotes=["*"],
        )

        self.connect("converter_relation.power_rel", "load_side.power")
        self.connect("converter_relation.voltage_out_rel", "generator_side.voltage_target")


class PerformancesRectifierTemperature(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.nonlinear_solver.options["stall_limit"] = 10
        self.nonlinear_solver.options["stall_tol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "resistance_profile",
            PerformancesResistance(number_of_points=number_of_points, rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "gate_voltage_profile",
            PerformancesGateVoltage(number_of_points=number_of_points, rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "conduction_losses",
            PerformancesConductionLosses(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "total_losses",
            PerformancesLosses(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "casing_temperature",
            PerformancesCasingTemperature(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "junction_temperature",
            PerformancesJunctionTemperature(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
