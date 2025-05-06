# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAirframeWeight(om.ExplicitComponent):
    """
    Computation of the mass of airframe.
    """

    def setup(self):
        self.add_input(
            "data:propulsion:he_power_train:mass",
            val=np.nan,
            units="kg",
            desc="Aircraft powertrain mass",
        )

        self.add_input(
            "data:weight:aircraft:OWE",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "data:cost:production:airframe:mass",
            val=2.0e3,
            units="kg",
            desc="Airframe mass of aircraft",
        )
        self.declare_partials("*", "data:weight:aircraft:OWE", val=1.0)
        self.declare_partials("*", "data:propulsion:he_power_train:mass", val=-1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:airframe:mass"] = (
            inputs["data:weight:aircraft:OWE"] - inputs["data:propulsion:he_power_train:mass"]
        )
