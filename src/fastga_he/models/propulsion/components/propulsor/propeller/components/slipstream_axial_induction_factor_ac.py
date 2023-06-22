# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerAxialInductionFactorWingAC(om.ExplicitComponent):
    """
    Computation of the axial induction factor at the wing aerodynamic center, implementation of
    the formula from :cite:`de:2019`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("axial_induction_factor", val=np.nan, shape=number_of_points)
        self.add_input(
            "contraction_ratio_squared",
            val=np.nan,
            shape=number_of_points,
            desc="Square of the contraction ratio of the propeller slipstream evaluated at the wing AC",
        )

        self.add_output(
            "axial_induction_factor_wing_ac",
            val=np.nan,
            shape=number_of_points,
            desc="Value of the axial induction factor at the wing aerodynamic chord",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        a_p = inputs["axial_induction_factor"]
        contraction_ratio_squared = inputs["contraction_ratio_squared"]

        a_w = (a_p + 1.0) / contraction_ratio_squared - 1.0

        outputs["axial_induction_factor_wing_ac"] = a_w

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        a_p = inputs["axial_induction_factor"]
        contraction_ratio_squared = inputs["contraction_ratio_squared"]

        partials["axial_induction_factor_wing_ac", "axial_induction_factor"] = np.diag(
            1.0 / contraction_ratio_squared
        )
        partials["axial_induction_factor_wing_ac", "contraction_ratio_squared"] = np.diag(
            -(a_p + 1.0) / contraction_ratio_squared ** 2.0
        )
