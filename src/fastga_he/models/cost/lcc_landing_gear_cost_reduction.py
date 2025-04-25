# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCLandingGearCostReduction(om.ExplicitComponent):
    """
    Computation of the landing gear cost reduction per aircraft from :cite:`gudmundsson:2013`.
    The cost of retractable landing gear is already include in other cost. Thus, the cost per
    aircraft needs to be adjusted as the fixed design is considered.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:fixed_landing_gear",
            val=np.nan,
            desc="Set to 1.0 if fixed, 0.0 for retractable landing gear",
        )

        self.add_output(
            "data:cost:production:landing_gear_cost_reduction",
            val=0.0,
            units="USD",
            desc="Cost reduction if fixed landing gear design is selected",
        )
        self.declare_partials("*", "*", val=-7500.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:landing_gear_cost_reduction"] = (
            -7500.0 * inputs["data:cost:production:fixed_landing_gear"]
        )
