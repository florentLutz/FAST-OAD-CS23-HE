# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCGeneratorOperation(om.ExplicitComponent):
    """
    Computation of the maintenance cost of the generator.
    """

    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        generator_id = self.options["generator_id"]

        self.add_input(
            "data:propulsion:he_power_train:generator:" + generator_id + ":cost_per_unit",
            units="USD",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the generator, typically around 15 year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":operation_cost",
            units="USD/yr",
            val=1e4,
            desc="Annual maintenance cost of a generator",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":operation_cost"] = (
            inputs["data:propulsion:he_power_train:generator:" + generator_id + ":cost_per_unit"]
            / inputs["data:propulsion:he_power_train:generator:" + generator_id + ":lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":operation_cost",
            "data:propulsion:he_power_train:generator:" + generator_id + ":cost_per_unit",
        ] = 1.0 / inputs["data:propulsion:he_power_train:generator:" + generator_id + ":lifespan"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":operation_cost",
            "data:propulsion:he_power_train:generator:" + generator_id + ":lifespan",
        ] = (
            -inputs["data:propulsion:he_power_train:generator:" + generator_id + ":cost_per_unit"]
            / inputs["data:propulsion:he_power_train:generator:" + generator_id + ":lifespan"]
            ** 2.0
        )
