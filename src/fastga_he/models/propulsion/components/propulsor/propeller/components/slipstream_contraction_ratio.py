# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerContractionRatio(om.ExplicitComponent):
    """
    Computation of the contraction ratio from the value of its square.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        self.add_input(
            "contraction_ratio_squared",
            val=np.nan,
            shape=number_of_points,
            desc="Square of the contraction ratio of the propeller slipstream evaluated at the wing AC",
        )

        self.add_output(
            "contraction_ratio",
            val=1.0,
            shape=number_of_points,
            desc="Contraction ratio of the propeller slipstream evaluated at the wing AC",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["contraction_ratio"] = np.sqrt(inputs["contraction_ratio_squared"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["contraction_ratio", "contraction_ratio_squared"] = 0.5 / np.sqrt(
            inputs["contraction_ratio_squared"]
        )
