# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCLearningCurveFactor(om.ExplicitComponent):
    """
    Computation of the aircraft production learning curve factor for tooling and manufacturing.
    The factor is obtained from http://www.ae.metu.edu.tr/~ae452sc2/lecture8_cost.pdf. The
    learning curve percentage falls between 80% to 90% based on the results from
    :cite:`bongers:2017`.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:learning_curve_percentage",
            val=85.0,
            units="percent",
            desc="Learning curve percentage for different discount rates on man hours required",
        )

        self.add_output(
            "data:cost:production:learning_curve_factor",
            val=0.765,
            desc="Learning curve factor for defining the man hours discount curve",
        )

        self.declare_partials(
            of="data:cost:production:learning_curve_factor",
            wrt="data:cost:production:learning_curve_percentage",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:learning_curve_factor"] = -5.64 + 1.44 * np.log(
            inputs["data:cost:production:learning_curve_percentage"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:cost:production:learning_curve_factor",
            "data:cost:production:learning_curve_percentage",
        ] = 1.44 / inputs["data:cost:production:learning_curve_percentage"]
