# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorHeight(om.ExplicitComponent):
    """
    Computation of the capacitor's height. Implementation of the workflow from
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
            name=prefix + ":capacitor:diameter",
            units="mm",
            val=np.nan,
            desc="Diameter of the capacitor",
        )
        self.add_input(
            name=prefix + ":capacitor:aspect_ratio",
            val=np.nan,
            desc="Ratio of the capacitor diameter over its height",
        )

        self.add_output(
            name=prefix + ":capacitor:height",
            units="mm",
            val=155,
            desc="Diameter of the capacitor",
        )

        self.declare_partials(
            of=prefix + ":capacitor:height",
            wrt=[prefix + ":capacitor:diameter", prefix + ":capacitor:aspect_ratio"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:height"] = (
            inputs[prefix + ":capacitor:aspect_ratio"] * inputs[prefix + ":capacitor:diameter"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        prefix = self.options["prefix"]

        partials[prefix + ":capacitor:height", prefix + ":capacitor:aspect_ratio"] = inputs[
            prefix + ":capacitor:diameter"
        ]
        partials[prefix + ":capacitor:height", prefix + ":capacitor:diameter"] = inputs[
            prefix + ":capacitor:aspect_ratio"
        ]
