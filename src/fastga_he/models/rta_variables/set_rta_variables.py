# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SetRTAVariable(om.ExplicitComponent):
    """
    Define variable that doesn't exist in FAST-OAD-RTA or requires complex computation.
    """

    def setup(self):
        self.add_input("data:TLAR:NPAX_design", val=np.nan)  # not used

        self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
        self.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="1/rad")
        self.add_output("data:aerodynamics:wing:cruise:CM0_clean")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:aerodynamics:horizontal_tail:cruise:CL0"] = -0.0068437669175491515

        outputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"] = 6.28

        outputs["data:aerodynamics:wing:cruise:CM0_clean"] = -0.02413516654351498
