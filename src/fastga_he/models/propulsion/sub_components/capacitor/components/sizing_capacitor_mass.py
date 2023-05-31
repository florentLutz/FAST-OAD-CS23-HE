# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorMass(om.ExplicitComponent):
    """
    Computation of the capacitor's mass. Implementation of the workflow from
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
            name="mass_ref",
            types=float,
            default=1.5,
            desc="Mass of the reference component [kg]",
        )

    def setup(self):

        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":capacitor:scaling:mass",
            val=np.nan,
            desc="Scaling factor for the capacitor mass",
        )

        self.add_output(
            name=prefix + ":capacitor:mass",
            units="kg",
            val=self.options["mass_ref"],
            desc="Capacitor's mass",
        )

        self.declare_partials(of="*", wrt="*", val=self.options["mass_ref"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:mass"] = (
            self.options["mass_ref"] * inputs[prefix + ":capacitor:scaling:mass"]
        )
