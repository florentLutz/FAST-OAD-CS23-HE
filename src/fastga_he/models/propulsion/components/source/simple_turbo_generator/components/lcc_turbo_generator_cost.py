# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO


import numpy as np
import openmdao.api as om


class LCCTurboGeneratorCost(om.ExplicitComponent):
    """
    Cost computation of the turbo generator based on the price of single starter generator from
    https://www.zauba.com/import-1152400-3-hs-code.html and the performance specification from
    https://www.startergenerator.com/inventory/1152400-3.
    """

    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):
        turbo_generator_id = self.options["turbo_generator_id"]

        self.add_input(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":shaft_power_rating",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":purchase_cost",
            units="USD",
            val=5.0e3,
            desc="Unit purchase cost of the turbo generator",
        )

        self.declare_partials(of="*", wrt="*", val=328.4)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]

        outputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":purchase_cost"
        ] = (
            328.4
            * inputs[
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":shaft_power_rating"
            ]
        )
