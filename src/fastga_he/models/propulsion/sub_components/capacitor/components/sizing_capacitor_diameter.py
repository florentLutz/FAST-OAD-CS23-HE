# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorDiameter(om.ExplicitComponent):
    """
    Computation of the capacitor's diameter. Implementation of the workflow from
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
            name="diameter_ref",
            types=float,
            default=100.0,
            desc="Diameter of the reference component [mm]",
        )

    def setup(self):

        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":capacitor:scaling:diameter",
            val=np.nan,
            desc="Scaling factor for the capacitor diameter",
        )

        self.add_output(
            name=prefix + ":capacitor:diameter",
            units="mm",
            val=self.options["diameter_ref"],
            desc="Diameter of the capacitor",
        )

        self.declare_partials(
            of=prefix + ":capacitor:diameter",
            wrt=prefix + ":capacitor:scaling:diameter",
            val=self.options["diameter_ref"],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:diameter"] = (
            self.options["diameter_ref"] * inputs[prefix + ":capacitor:scaling:diameter"]
        )
