# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga_he.models.performances.op_mission_vector.op_mission_vector import (
    OperationalMissionVector,
)


class OperationalMissionVectorWithTargetFuel(OperationalMissionVector):
    def setup(self):
        super().setup()
        self.add_subsystem(name="distance_to_target", subsys=DistanceToTargetFuel(), promotes=["*"])


class DistanceToTargetFuel(om.ImplicitComponent):
    def setup(self):
        self.add_input("data:mission:operational:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:payload_range:target_fuel", val=np.nan, units="kg")

        self.add_output("data:mission:operational:range", units="NM", val=30.0)

        self.declare_partials(
            of="data:mission:operational:range", wrt="data:mission:operational:fuel", val=1.0
        )
        self.declare_partials(
            of="data:mission:operational:range",
            wrt="data:mission:payload_range:target_fuel",
            val=-1.0,
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        residuals["data:mission:operational:range"] = (
            inputs["data:mission:operational:fuel"]
            - inputs["data:mission:payload_range:target_fuel"]
        )
