# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class LCCSpeedReducerCost(om.ExplicitComponent):
    """
    Computation of speed reducer purchase price, which is based on the gearbox provided
    by https://www.mohawkaero.com/product-page/ak7-gearbox. The gearbox weight is obtained
    from https://www.youtube.com/watch?v=M10O7S89GE8.
    """

    def initialize(self):
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )

    def setup(self):
        speed_reducer_id = self.options["speed_reducer_id"]

        self.add_input(
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the speed reducer",
        )

        self.add_output(
            name="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":purchase_cost",
            val=1.0e3,
            units="USD",
        )

        self.declare_partials(of="*", wrt="*", val=230.98)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]

        outputs[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":purchase_cost"
        ] = (
            230.98
            * inputs["data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":mass"]
        )
