# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SlipstreamPropellerDeltaCM(om.ExplicitComponent):
    """
    Sums the contribution of the increase of dynamic pressure and increase in lift on the
    pitching moment as detailed in :cite:`bouquet:2017`. The second term was neglected.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "delta_Cm0",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in the profile pitching moment coefficient downstream of the propeller",
        )
        self.add_input(
            "delta_Cm_alpha",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in pitching moment due to lift caused by the propeller slipstream",
        )

        self.add_output(
            "delta_Cm",
            val=-0.01,
            shape=number_of_points,
            desc="Increase in the pitching moment coefficient downstream of the propeller",
        )

        self.declare_partials(
            of="delta_Cm",
            wrt="delta_Cm0",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )
        self.declare_partials(
            of="delta_Cm",
            wrt="delta_Cm_alpha",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["delta_Cm"] = inputs["delta_Cm0"] + inputs["delta_Cm_alpha"]
