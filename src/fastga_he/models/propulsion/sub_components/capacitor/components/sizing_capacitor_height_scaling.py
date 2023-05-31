# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorHeightScaling(om.ExplicitComponent):
    """
    Computation of the scaling of the capacitor's height. Implementation of the workflow from
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
            name="height_ref",
            types=float,
            default=155.0,
            desc="Height of the reference component [mm]",
        )

    def setup(self):

        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":capacitor:height",
            units="mm",
            val=np.nan,
            desc="Diameter of the capacitor",
        )

        self.add_output(
            name=prefix + ":capacitor:scaling:height",
            val=1.0,
            desc="Scaling factor for the capacitor height",
        )

        self.declare_partials(of="*", wrt="*", val=1.0 / self.options["height_ref"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:scaling:height"] = (
            inputs[prefix + ":capacitor:height"] / self.options["height_ref"]
        )
