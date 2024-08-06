# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class UpdateTOW(om.ExplicitComponent):
    """
    Computes the TOW of the operational mission. Will most likely create a loop since he fuel
    consumed will depend on the aircraft weight which depends on fuel consumed. It was decided to
    have it here rather than in the mass as the update_mtow component as it will depend on the
    choice of operational mission
    """

    def setup(self):
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:mission:operational:payload:mass", val=np.nan, units="kg")
        self.add_input("data:mission:operational:fuel", val=np.nan, units="kg")

        self.add_output("data:mission:operational:TOW", val=1400.0, units="kg")

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:mission:operational:TOW"] = (
            inputs["data:weight:aircraft:OWE"]
            + inputs["data:mission:operational:payload:mass"]
            + inputs["data:mission:operational:fuel"]
        )
