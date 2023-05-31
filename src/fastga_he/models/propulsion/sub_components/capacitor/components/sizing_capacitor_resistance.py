# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorResistance(om.ExplicitComponent):
    """
    Computation of the capacitor's resistance. Implementation of the workflow from
    :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a capacitor",
            allow_none=False,
        )
        self.options.declare(
            name="r_ref",
            types=float,
            default=3.2e-3,
            desc="Resistance of the reference component [ohm]",
        )

    def setup(self):

        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":capacitor:scaling:resistance",
            val=np.nan,
            desc="Scaling factor for the capacitor resistance",
        )

        self.add_output(
            name=prefix + ":capacitor:resistance",
            units="ohm",
            val=self.options["r_ref"],
            desc="Capacitor's resistance",
        )

        self.declare_partials(of="*", wrt="*", val=self.options["r_ref"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:resistance"] = (
            self.options["r_ref"] * inputs[prefix + ":capacitor:scaling:resistance"]
        )
