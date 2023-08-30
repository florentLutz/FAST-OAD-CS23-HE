# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .perf_current import PerformancesCurrent, PerformancesHarnessCurrent
from .perf_losses_one_cable import PerformancesLossesOneCable
from .perf_maximum import PerformancesMaximum

from ..constants import (
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE,
    SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE,
)
from ..components.perf_temperature_constant import SUBMODEL_DC_LINE_TEMPERATURE_CONSTANT


class PerformancesHarness(om.Group):
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
            "temperature_profile",
            PerformancesHarnessTemperature(
                harness_id=harness_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        # Though harness current depend on a variable stuck in the loop its output is not used in
        # the loop so we can take it out
        self.add_subsystem(
            "harness_current",
            PerformancesHarnessCurrent(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(harness_id=harness_id, number_of_points=number_of_points),
            promotes=[
                "*",
            ],
        )


class PerformancesHarnessTemperature(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # In case the temperature is constant we don't need this solver hence this check. It must
        # be noted however that it does not appear to significantly reduce computation time,
        # quite the contrary ...
        if (
            oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE]
            != SUBMODEL_DC_LINE_TEMPERATURE_CONSTANT
        ):

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
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        options_dict = {"harness_id": harness_id, "number_of_points": number_of_points}

        # Change of plans, we put the component that outputs the temperature at the very
        # beginning, this way when the temperature is fixed, it does not do an unnecessary loop
        # and when it is not it does not increase too much the time it takes to run
        self.add_subsystem(
            "temperature",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE, options=options_dict
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "resistance",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE, options=options_dict
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "cable_current",
            PerformancesCurrent(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "losses_one_cable",
            PerformancesLossesOneCable(number_of_points=number_of_points),
            promotes=["*"],
        )
