# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

KILMER = np.nan


class PerformancesEfficiency(om.ExplicitComponent):
    """
    Computation of the efficiency of the propeller, which does not directly serve any purpose in
    the computation but is there to show the figures of merit of the propulsion chain.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("thrust_coefficient", val=KILMER, shape=number_of_points)
        self.add_input("power_coefficient", val=KILMER, shape=number_of_points)
        self.add_input("advance_ratio", val=KILMER, shape=number_of_points)

        self.add_output("efficiency", shape=number_of_points, lower=0.0, upper=1.0, val=0.8)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        j = inputs["advance_ratio"]
        ct = inputs["thrust_coefficient"]
        cp = inputs["power_coefficient"]

        # When the propeller is not used (0W of power on the shaft, the efficiency is set to 1.0)
        efficiency = np.ones_like(j)
        np.divide(j * ct, cp, out=efficiency, where=cp != 0)
        outputs["efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        j = inputs["advance_ratio"]
        ct = inputs["thrust_coefficient"]
        cp = inputs["power_coefficient"]

        partials_advance_ratio = np.full_like(j, 1e-6)
        np.divide(ct, cp, out=partials_advance_ratio, where=cp != 0)
        partials["efficiency", "advance_ratio"] = partials_advance_ratio

        partials_ct = np.full_like(ct, 1e-6)
        np.divide(j, cp, out=partials_ct, where=cp != 0)
        partials["efficiency", "thrust_coefficient"] = partials_ct

        partials_cp = np.full_like(cp, 1e-6)
        np.divide(-j * ct, cp**2.0, out=partials_cp, where=cp != 0)
        partials["efficiency", "power_coefficient"] = partials_cp
