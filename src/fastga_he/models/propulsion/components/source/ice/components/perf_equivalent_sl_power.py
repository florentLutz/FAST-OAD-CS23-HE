# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from stdatm import Atmosphere


class PerformancesEquivalentSeaLevelPower(om.ExplicitComponent):
    """
    Because ICE tend to lose maximum power with altitude, and because we want to make sure to
    size the component (or at least make sure he can do what we ask of him), we will here compute
    what would be the maximum SL power required to achieve the power required. The formula used
    is the so-called Gagg and Ferrar model. Formula taken from :cite:`gudmundsson:2013`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input("altitude", units="m", val=0.0, shape=number_of_points)

        self.add_output("equivalent_SL_power", units="W", val=np.full(number_of_points, 250e3))

        self.declare_partials(of="equivalent_SL_power", wrt="shaft_power_out", method="exact")
        self.declare_partials(
            of="equivalent_SL_power",
            wrt="altitude",
            method="fd",
            form="central",
            step=1.0e2,
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        shaft_power_out = inputs["shaft_power_out"]

        rho = Atmosphere(inputs["altitude"], altitude_in_feet=False).density
        sigma = rho / Atmosphere(0.0).density

        corrective_factor = sigma - (1 - sigma) / 7.55

        outputs["equivalent_SL_power"] = shaft_power_out / corrective_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rho = Atmosphere(inputs["altitude"], altitude_in_feet=False).density
        sigma = rho / Atmosphere(0.0).density

        corrective_factor = sigma - (1 - sigma) / 7.55

        partials["equivalent_SL_power", "shaft_power_out"] = np.diag(1.0 / corrective_factor)
