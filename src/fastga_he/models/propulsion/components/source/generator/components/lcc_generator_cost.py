# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCGeneratorCost(om.ExplicitComponent):
    """
    Computation of the cost of the generator. This is based on the PMSM regression model.
    """

    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        generator_id = self.options["generator_id"]

        self.add_input(
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":cost_per_unit",
            units="USD",
            val=1e4,
            desc="Cost of the generator per unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max"
        ]

        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":cost_per_unit"] = (
            1876.1 * np.exp(0.0062 * power_max)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max"
        ]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":cost_per_unit",
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max",
        ] = 11.632 * np.exp(0.0062 * power_max)
