# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerAxialInductionFactor(om.ExplicitComponent):
    """
    Adaptation of the formula taken from :cite:`de:2019` for the computation of the axial
    induction factor.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("thrust_loading", val=np.nan, shape=number_of_points)

        self.add_output("axial_induction_factor", val=0.1, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        t_c = inputs["thrust_loading"]

        a_p = 0.5 * (np.sqrt(1.0 + 8.0 / np.pi * t_c) - 1.0)

        outputs["axial_induction_factor"] = a_p

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # To avoid unwanted division by 0
        t_c = np.clip(inputs["thrust_loading"], 1e-6, None)

        partials["axial_induction_factor", "thrust_loading"] = np.diag(
            1.0 / np.sqrt(1.0 + 8.0 / np.pi * t_c) * 2.0 / np.pi
        )
