# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerDeltaCl(om.ExplicitComponent):
    """
    Compute the increase in lift coefficient due to the blowing of the wing. Considers the 2D
    effect and a blown area ratio to transpose to 3D.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "delta_Cl_2D",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in the section lift downstream of the propeller",
        )
        self.add_input(
            "blown_area_ratio",
            val=np.nan,
            shape=number_of_points,
            desc="Portion of the wing blown by the propeller",
        )

        self.add_output(
            "delta_Cl",
            val=0.01,
            shape=number_of_points,
            desc="Increase in the lift coefficient downstream of the propeller",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["delta_Cl"] = inputs["delta_Cl_2D"] * inputs["blown_area_ratio"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["delta_Cl", "delta_Cl_2D"] = np.diag(inputs["blown_area_ratio"])
        partials["delta_Cl", "blown_area_ratio"] = np.diag(inputs["delta_Cl_2D"])
