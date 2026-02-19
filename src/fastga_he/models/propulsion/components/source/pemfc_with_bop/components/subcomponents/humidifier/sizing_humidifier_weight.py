# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHumidifierWeight(om.ExplicitComponent):
    """
    Weight computation of the humidifier.
    """

    def setup(self):
        ## Inputs
        self.add_input(name="data:thermal:fuel_cell:power", units="kW", val=np.nan)

        ## Output
        self.add_output(name="data:thermal:humidifier:mass", units="kg")
        self.add_output(name="data:thermal:humidifier:volume", units="m**3")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ## Inputs
        P_fc = inputs["data:thermal:fuel_cell:power"]

        if P_fc <= 200:
            M = 4.535e-5 * P_fc**2 + 0.062 * P_fc + 2.119
        else:
            M = 16.33 * (P_fc / 200)

        if P_fc <= 200:
            V = (-4.058 * 1e-4 * P_fc**2 + 0.151 * P_fc + 2.019) / 1e3
        else:
            V = 15.54 * (P_fc / 150) / 1e3

        ## Outputs
        outputs["data:thermal:humidifier:mass"] = M
        outputs["data:thermal:humidifier:volume"] = V
