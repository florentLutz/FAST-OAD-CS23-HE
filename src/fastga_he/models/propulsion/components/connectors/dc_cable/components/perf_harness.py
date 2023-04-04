# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .perf_resistance import PerformancesResistance
from .perf_current import PerformancesCurrent, PerformancesHarnessCurrent
from .perf_losses_one_cable import PerformancesLossesOneCable
from .perf_maximum import PerformancesMaximum

from ..constants import SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE


class PerformancesHarness(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.nonlinear_solver.options["stall_limit"] = 5
        self.nonlinear_solver.options["stall_tol"] = 1e-5
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.linear_solver = om.LinearBlockGS()

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
            promotes=["data:*", "settings:*", "cable_temperature"],
        )
        self.add_subsystem(
            "cable_current",
            PerformancesCurrent(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["data:*", "dc_voltage_out", "dc_voltage_in"],
        )
        self.add_subsystem(
            "losses_one_cable",
            PerformancesLossesOneCable(number_of_points=number_of_points),
            promotes=["conduction_losses"],
        )

        options_dict = {"harness_id": harness_id, "number_of_points": number_of_points}

        self.add_subsystem(
            "temperature",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE, options=options_dict
            ),
            promotes=[
                "data:*",
                "exterior_temperature",
                "cable_temperature",
                "conduction_losses",
                "time_step",
            ],
        )
        # Though harness current depend on a variable stuck in the loop its output is not used in
        # the loop so we can take it out
        self.add_subsystem(
            "harness_current",
            PerformancesHarnessCurrent(harness_id=harness_id, number_of_points=number_of_points),
            promotes=["data:*", "dc_current"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(harness_id=harness_id, number_of_points=number_of_points),
            promotes=[
                "data:*",
                "dc_voltage_out",
                "dc_voltage_in",
                "cable_temperature",
                "conduction_losses",
            ],
        )

        self.connect(
            "resistance.resistance_per_cable",
            ["cable_current.resistance_per_cable", "losses_one_cable.resistance_per_cable"],
        )

        self.connect(
            "cable_current.dc_current_one_cable",
            [
                "harness_current.dc_current_one_cable",
                "losses_one_cable.dc_current_one_cable",
                "maximum.dc_current_one_cable",
            ],
        )
