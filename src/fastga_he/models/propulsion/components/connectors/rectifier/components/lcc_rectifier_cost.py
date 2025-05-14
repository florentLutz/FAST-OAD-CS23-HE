# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCRectifierCost(om.ExplicitComponent):
    """
    Computation of the rectifier purchase cost. The retail price is provided by
    https://www.ato.com/plating-rectifier.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max",
            units="A",
            val=np.nan,
            desc="Maximum RMS current flowing through one arm of the rectifier",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":cost_per_unit",
            units="USD",
            val=3500.0,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            val=1.72,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":cost_per_unit"] = (
            1.72
            * inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max"]
            + 2034.0
        )
