# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCLandingGearCostReduction(om.ExplicitComponent):
    """
    Per :cite:`gudmundsson:2013`, the cost of the aircraft assumes a retractable landing. To account for fixed landing gear, a cost reduction is recommenced.
    """

    def setup(self):
        self.add_input(
            "data:geometry:landing_gear:type",
            val=np.nan,
            desc="Set to 0.0 if fixed, 1.0 for retractable landing gear",
        )

        self.add_output(
            "data:cost:production:landing_gear_cost_reduction",
            val=0.0,
            units="USD",
            desc="Cost reduction if fixed landing gear design is selected",
        )
        self.declare_partials(
            of="data:cost:production:landing_gear_cost_reduction",
            wrt="data:geometry:landing_gear:type",
            val=7500.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:landing_gear_cost_reduction"] = 7500.0 * (
            inputs["data:geometry:landing_gear:type"] - 1.0
        )
