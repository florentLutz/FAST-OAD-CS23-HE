# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorMass(om.ExplicitComponent):
    """
    Computation of the mass of the filter inductor, implementation in the openMDAO format of
    :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )

    def setup(self):

        prefix = self.options["prefix"]

        self.add_input(
            prefix + ":inductor:core_mass",
            units="kg",
            val=np.nan,
            desc="Mass of the E-core in the inductor",
        )
        self.add_input(
            name=prefix + ":inductor:copper_mass",
            units="kg",
            val=np.nan,
            desc="Mass of the copper in the inductor",
        )

        self.add_output(
            name=prefix + ":inductor:mass",
            units="kg",
            val=1.0,
            desc="Mass of the inductor",
        )

        self.declare_partials(
            of=prefix + ":inductor:mass",
            wrt=prefix + ":inductor:copper_mass",
            val=1.0,
        )

        self.declare_partials(
            of=prefix + ":inductor:mass",
            wrt=prefix + ":inductor:core_mass",
            val=2.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":inductor:mass"] = (
            inputs[prefix + ":inductor:copper_mass"] + 2.0 * inputs[prefix + ":inductor:core_mass"]
        )
