# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorDiameterScaling(om.ExplicitComponent):
    """
    Computation of the scaling of the capacitor's diameter. Implementation of the workflow from
    :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a capacitor",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":capacitor:scaling:capacity",
            val=np.nan,
            desc="Scaling factor for the capacitor capacity",
        )
        self.add_input(
            name=prefix + ":capacitor:aspect_ratio",
            val=1.0,
            desc="Ratio of the capacitor diameter over its height",
        )

        self.add_output(
            name=prefix + ":capacitor:scaling:diameter",
            val=1.0,
            desc="Scaling factor for the capacitor diameter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        capacity_star = inputs[prefix + ":capacitor:scaling:capacity"]
        aspect_ratio = inputs[prefix + ":capacitor:aspect_ratio"]

        outputs[prefix + ":capacitor:scaling:diameter"] = (capacity_star * aspect_ratio) ** (
            1.0 / 3.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        capacity_star = inputs[prefix + ":capacitor:scaling:capacity"]
        aspect_ratio = inputs[prefix + ":capacitor:aspect_ratio"]

        partials[prefix + ":capacitor:scaling:diameter", prefix + ":capacitor:scaling:capacity"] = (
            1.0 / 3.0 * (capacity_star**-2.0 * aspect_ratio) ** (1.0 / 3.0)
        )
        partials[prefix + ":capacitor:scaling:diameter", prefix + ":capacitor:aspect_ratio"] = (
            1.0 / 3.0 * (capacity_star * aspect_ratio**-2.0) ** (1.0 / 3.0)
        )
