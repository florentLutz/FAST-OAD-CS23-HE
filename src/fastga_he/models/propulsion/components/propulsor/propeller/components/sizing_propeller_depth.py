# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPropellerDepth(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="cm",
            desc="Diameter of the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":depth_to_diameter_ratio",
            val=0.15,
            desc="Ratio between the propeller depth and propeller diameter, default at 0.15",
        )

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":depth",
            val=0.3,
            units="cm",
            desc="Depth of the propeller",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":depth"] = (
            inputs[
                "data:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":depth_to_diameter_ratio"
            ]
            * inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":depth",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":depth_to_diameter_ratio",
        ] = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":depth",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":depth_to_diameter_ratio"
        ]
