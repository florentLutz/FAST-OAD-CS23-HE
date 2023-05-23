# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRectifierCapacitorWeight(om.ExplicitComponent):
    """
    Computation of the weight of the capacitor, regression based on capacitors from AVX,
    does not take rms current into account but it is an important criterion when choosing data
    from a catalog.
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
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity",
            val=np.nan,
            units="F",
            desc="Capacity required to dampen the design voltage ripple",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:mass",
            val=5000.0,
            units="g",
            desc="Mass of the capacitor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]

        # We need the capacity in microF
        capacity = (
            inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity"
            ]
            * 1e6
        )

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:mass"] = (
            10.524 * capacity ** 0.7749
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rectifier_id = self.options["rectifier_id"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity",
        ] = (
            10.524
            * 1e6 ** 0.7749
            * 0.7749
            * inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity"
            ]
            ** (0.7749 - 1.0)
        )
