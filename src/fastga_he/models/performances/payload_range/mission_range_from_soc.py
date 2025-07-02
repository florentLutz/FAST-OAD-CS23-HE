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
    "fastga_he.performances.op_mission_target_SoC", domain=ModelDomain.OTHER
)
class OperationalMissionVectorWithTargetSoC(OperationalMissionVector):
    """
    Allows to compute the mission but instead of setting the target range to achieve, have the
    aircraft fly as long as there is energy in the battery. Actually, have the aircraft fly as long
    as the state of charge of a battery is below the target threshold.
    """

    def initialize(self):
        super().initialize()

        self.options.declare(
            "variable_name_target_SoC",
            types=str,
            default=None,
            allow_none=False,
            desc="Name of the variable that will be used to evaluate if target SOC is reached, "
            "':' should be replaced by '__'",
        )
        self.options.declare(
            "variable_name_threshold_SoC",
            types=str,
            default="data:mission:operational:threshold_SoC",
            allow_none=False,
            desc="Name of the variable that contains the target SOC, ':' should be replaced by "
            "'__'",
        )

    def setup(self):
        super().setup()

        variable_name_target_soc = self.options["variable_name_target_SoC"].replace("__", ":")
        variable_name_threshold_soc = self.options["variable_name_threshold_SoC"].replace("__", ":")

        self.add_subsystem(
            name="distance_to_target",
            subsys=DistanceToTargetSoc(
                variable_name_target_SoC=variable_name_target_soc,
                variable_name_threshold_SoC=variable_name_threshold_soc,
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


class DistanceToTargetSoc(om.ImplicitComponent):
    def initialize(self):
        self.options.declare(
            "variable_name_target_SoC",
            types=str,
            default=None,
            allow_none=False,
            desc="Name of the variable that will be used to evaluate if target SOC is reached",
        )
        self.options.declare(
            "variable_name_threshold_SoC",
            types=str,
            default="data:mission:operational:threshold_SoC",
            allow_none=False,
            desc="Name of the variable that contains the target SOC",
        )

    def setup(self):
        variable_name_target_soc = self.options["variable_name_target_SoC"]
        variable_name_threshold_soc = self.options["variable_name_threshold_SoC"]

        self.add_input(variable_name_target_soc, val=np.nan, units="percent")
        self.add_input(variable_name_threshold_soc, val=np.nan, units="percent")

        self.add_output("data:mission:operational:range", units="NM", val=30.0)

        self.declare_partials(
            of="data:mission:operational:range", wrt=variable_name_target_soc, val=1.0
        )
        self.declare_partials(
            of="data:mission:operational:range",
            wrt=variable_name_threshold_soc,
            val=-1.0,
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        variable_name_target_soc = self.options["variable_name_target_SoC"]
        variable_name_threshold_soc = self.options["variable_name_threshold_SoC"]

        residuals["data:mission:operational:range"] = (
            inputs[variable_name_target_soc] - inputs[variable_name_threshold_soc]
        )
