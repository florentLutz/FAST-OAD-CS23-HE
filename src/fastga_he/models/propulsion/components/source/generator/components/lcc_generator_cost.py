# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCGeneratorCost(om.ExplicitComponent):
    """
    Computation of the generator purchase cost. This is based on the PMSM regression model.
    """

    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        generator_id = self.options["generator_id"]

        self.add_input(
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_rating",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":purchase_cost",
            units="USD",
            val=1e4,
            desc="Unit purchase cost of the generator",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_rating"
        ]

        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":purchase_cost"] = (
            893.51 * np.exp(0.0281 * power_rating)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_rating"
        ]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":purchase_cost",
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_rating",
        ] = 25.108 * np.exp(0.0281 * power_rating)
