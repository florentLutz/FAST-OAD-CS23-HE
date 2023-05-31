# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorResistanceScaling(om.ExplicitComponent):
    """
    Computation of the scaling of the capacitor's resistance. Implementation of the workflow from
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
            name=prefix + ":capacitor:scaling:diameter",
            val=np.nan,
            desc="Scaling factor for the capacitor diameter",
        )

        self.add_output(
            name=prefix + ":capacitor:scaling:resistance",
            val=1.0,
            desc="Scaling factor for the capacitor resistance",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:scaling:resistance"] = (
            inputs[prefix + ":capacitor:scaling:diameter"] ** -2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        prefix = self.options["prefix"]

        partials[
            prefix + ":capacitor:scaling:resistance", prefix + ":capacitor:scaling:diameter"
        ] = (-2.0 * inputs[prefix + ":capacitor:scaling:diameter"] ** -3.0)
