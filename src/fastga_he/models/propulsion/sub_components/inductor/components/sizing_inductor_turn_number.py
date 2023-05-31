# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorTurnNumber(om.ExplicitComponent):
    """
    Computation of the number of turns in the filter inductor, implementation of the formula from
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
            name=prefix + ":inductor:reluctance",
            units="H**-1",
            val=np.nan,
            desc="Reluctance of the inductor",
        )
        self.add_input(
            name=prefix + ":inductor:inductance",
            units="H",
            val=np.nan,
            desc="Inductance of the inductor",
        )

        self.add_output(
            name=prefix + ":inductor:turn_number",
            val=20.0,
            desc="Number of turns in the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":inductor:turn_number"] = np.sqrt(
            inputs[prefix + ":inductor:inductance"] * inputs[prefix + ":inductor:reluctance"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        prefix = self.options["prefix"]

        partials[prefix + ":inductor:turn_number", prefix + ":inductor:inductance"] = 0.5 * np.sqrt(
            inputs[prefix + ":inductor:reluctance"] / inputs[prefix + ":inductor:inductance"]
        )
        partials[prefix + ":inductor:turn_number", prefix + ":inductor:reluctance"] = 0.5 * np.sqrt(
            inputs[prefix + ":inductor:inductance"] / inputs[prefix + ":inductor:reluctance"]
        )
