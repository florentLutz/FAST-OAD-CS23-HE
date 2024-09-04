# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere

DENSITY_SL = Atmosphere(0.0).density


class SlipstreamDensityRatio(om.ExplicitComponent):
    """
    Computation of the density ratio, often called sigma in literature used in the surrogate. Same
    as the one in the performances but since both group don't dialogue, it needs to be recomputed.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)

        self.add_output("density_ratio", val=1.0, shape=number_of_points, lower=0.0)

        self.declare_partials(
            of="*",
            wrt="*",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.full(number_of_points, 1.0 / DENSITY_SL),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["density_ratio"] = inputs["density"] / DENSITY_SL
