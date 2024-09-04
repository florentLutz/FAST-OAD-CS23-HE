# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga_he.models.performances.op_mission_vector.op_mission_vector import (
    OperationalMissionVector,
)


class OperationalMissionVectorWithTargetSoC(OperationalMissionVector):
    def initialize(self):
        super().initialize()

        self.options.declare(
            "variable_name_target_SoC",
            types=str,
            default=None,
            allow_none=False,
            desc="Name of the variable that will be used to evaluate if target SOC is reached",
        )

    def setup(self):
        super().setup()
        self.add_subsystem(
            name="distance_to_target",
            subsys=DistanceToTargetSoc(
                variable_name_target_SoC=self.options["variable_name_target_SoC"]
            ),
            promotes=["*"],
        )


class DistanceToTargetSoc(om.ImplicitComponent):
    def initialize(self):
        self.options.declare(
            "variable_name_target_SoC",
            types=str,
            default=None,
            allow_none=False,
            desc="Name of the variable that will be used to evaluate if target SOC is reached",
        )

    def setup(self):
        variable_name_target_soc = self.options["variable_name_target_SoC"]

        self.add_input(variable_name_target_soc, val=np.nan, units="percent")
        self.add_input("data:mission:payload_range:threshold_SoC", val=np.nan, units="percent")

        self.add_output("data:mission:operational:range", units="NM", val=30.0)

        self.declare_partials(
            of="data:mission:operational:range", wrt=variable_name_target_soc, val=1.0
        )
        self.declare_partials(
            of="data:mission:operational:range",
            wrt="data:mission:payload_range:threshold_SoC",
            val=-1.0,
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        variable_name_target_soc = self.options["variable_name_target_SoC"]

        residuals["data:mission:operational:range"] = (
            inputs[variable_name_target_soc] - inputs["data:mission:payload_range:threshold_SoC"]
        )
