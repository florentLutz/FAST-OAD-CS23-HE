# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_resistance import PerformancesResistance
from .perf_current import PerformancesCurrent, PerformancesHarnessCurrent
from .perf_temperature_derivative import PerformancesTemperatureDerivative
from .perf_temperature_increase import PerformancesTemperatureIncrease
from .perf_temperature import PerformancesTemperature
from .perf_maximum import PerformancesMaximum


class PerformanceHarness(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "resistance",
            PerformancesResistance(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["data:*", "settings:*"],
        )
        self.add_subsystem(
            "cable_current",
            PerformancesCurrent(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["data:*", "voltage_a", "voltage_b"],
        )
        self.add_subsystem(
            "harness_current",
            PerformancesHarnessCurrent(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["data:*", "total_current"],
        )
        self.add_subsystem(
            "temperature_derivative",
            PerformancesTemperatureDerivative(
                harness_id=harness_id, number_of_points=number_of_points
            ),
            promotes=["data:*", "exterior_temperature"],
        )
        self.add_subsystem(
            "temperature_increase",
            PerformancesTemperatureIncrease(number_of_points=number_of_points),
            promotes=["time_step"],
        )
        self.add_subsystem(
            "temperature",
            PerformancesTemperature(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "maxima",
            PerformancesMaximum(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["data:*", "total_current", "voltage_a", "voltage_b"],
        )

        self.connect(
            "resistance.resistance_per_cable",
            ["cable_current.resistance_per_cable", "temperature_derivative.resistance_per_cable"],
        )

        self.connect(
            "cable_current.current", ["harness_current.current", "temperature_derivative.current"]
        )

        self.connect(
            "temperature_derivative.cable_temperature_time_derivative",
            "temperature_increase.cable_temperature_time_derivative",
        )

        self.connect(
            "temperature_increase.cable_temperature_increase",
            "temperature.cable_temperature_increase",
        )

        self.connect(
            "temperature.cable_temperature",
            [
                "resistance.cable_temperature",
                "temperature_derivative.cable_temperature",
                "maxima.cable_temperature",
            ],
        )
