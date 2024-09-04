# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorCapacityScaling(om.ExplicitComponent):
    """
    Computation of the scaling of the capacitor's capacity. Implementation of the workflow from
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
            name="capacity_ref",
            types=float,
            default=1000.0,
            desc="Capacity of the reference component [uF]",
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":capacitor:capacity",
            units="uF",
            val=np.nan,
            desc="Capacity of the capacitor",
        )

        self.add_output(
            name=prefix + ":capacitor:scaling:capacity",
            val=1.0,
            desc="Scaling factor for the capacitor capacity",
        )

        self.declare_partials(of="*", wrt="*", val=1.0 / self.options["capacity_ref"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:scaling:capacity"] = (
            inputs[prefix + ":capacitor:capacity"] / self.options["capacity_ref"]
        )
