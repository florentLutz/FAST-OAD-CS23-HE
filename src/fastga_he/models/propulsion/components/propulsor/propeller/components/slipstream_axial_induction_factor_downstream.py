# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerVelocityRatioDownstream(om.ExplicitComponent):
    """
    Computation of the velocity ratio far downstream of the propeller, result form the ADT.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("axial_induction_factor", val=np.nan, shape=number_of_points)

        self.add_output("velocity_ratio_downstream", val=1.25, shape=number_of_points)

        self.declare_partials(
            of="velocity_ratio_downstream",
            wrt="axial_induction_factor",
            val=2.0 * np.eye(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["velocity_ratio_downstream"] = 1.0 + 2.0 * inputs["axial_induction_factor"]
