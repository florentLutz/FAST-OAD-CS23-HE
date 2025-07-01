# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga_he.models.performances.op_mission_vector.op_mission_vector import (
    OperationalMissionVector,
)


@oad.RegisterOpenMDAOSystem(
    "fastga_he.performances.op_mission_target_fuel", domain=ModelDomain.OTHER
)
class OperationalMissionVectorWithTargetFuel(OperationalMissionVector):
    """
    Allows to compute the mission but instead of setting the target range to achieve, have the
    aircraft fly as long as there is fuel in the tanks. Actually, have the aircraft fly as long
    as the fuel in one of the tanks is below a certain value
    """

    def initialize(self):
        super().initialize()

        self.options.declare(
            "variable_name_threshold_fuel",
            types=str,
            default="data:mission:operational:threshold_fuel",
            allow_none=False,
            desc="Name of the variable that contains the target fuel",
        )

    def setup(self):
        super().setup()
        self.add_subsystem(
            name="distance_to_target",
            subsys=DistanceToTargetFuel(
                variable_name_threshold_fuel=self.options["variable_name_threshold_fuel"]
            ),
            promotes=["*"],
        )

        # Replace the old solver with a NewtonSolver to handle the ImplicitComponent
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.nonlinear_solver.options["atol"] = 1e-5
        self.nonlinear_solver.options["stall_limit"] = 2
        self.nonlinear_solver.options["stall_tol"] = 1e-5
        self.linear_solver = om.DirectSolver()


class DistanceToTargetFuel(om.ImplicitComponent):
    def initialize(self):
        self.options.declare(
            "variable_name_threshold_fuel",
            types=str,
            default="data:mission:operational:threshold_fuel",
            allow_none=False,
            desc="Name of the variable that contains the target fuel",
        )

    def setup(self):
        variable_name_threshold_fuel = self.options["variable_name_threshold_fuel"]

        self.add_input("data:mission:operational:fuel", val=np.nan, units="kg")
        self.add_input(variable_name_threshold_fuel, val=np.nan, units="kg")

        self.add_output("data:mission:operational:range", units="NM", val=30.0)

        self.declare_partials(
            of="data:mission:operational:range", wrt="data:mission:operational:fuel", val=1.0
        )
        self.declare_partials(
            of="data:mission:operational:range",
            wrt=variable_name_threshold_fuel,
            val=-1.0,
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        variable_name_threshold_fuel = self.options["variable_name_threshold_fuel"]
        residuals["data:mission:operational:range"] = (
            inputs["data:mission:operational:fuel"] - inputs[variable_name_threshold_fuel]
        )
