# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAWingWeightPerFU(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:environmental_impact:aircraft_per_fu", val=np.nan)

        self.add_output("data:weight:airframe:wing:mass_per_fu", val=1e-6, units="kg")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:weight:airframe:wing:mass_per_fu"] = (
            inputs["data:weight:airframe:wing:mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:weight:airframe:wing:mass_per_fu", "data:weight:airframe:wing:mass"] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
        )
        partials[
            "data:weight:airframe:wing:mass_per_fu", "data:environmental_impact:aircraft_per_fu"
        ] = inputs["data:weight:airframe:wing:mass"]
