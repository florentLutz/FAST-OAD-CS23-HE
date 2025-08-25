# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain
from stdatm import AtmosphereWithPartials


@oad.RegisterOpenMDAOSystem("fastga_he.correct_input.aero_approx2", domain=ModelDomain.GEOMETRY)
class Aero_compute2(om.ExplicitComponent):
    " Computation of the cl_ref based on an elliptic distribution assumption"
    def setup(self):
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:oswald_coefficient", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:b_50", val=np.nan, units="m")

        self.add_output("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient")
        self.add_output("data:aerodynamics:wing:low_speed:CL_ref")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        c_root = inputs["data:geometry:wing:root:chord"].item()
        c_tip = inputs["data:geometry:wing:tip:chord"].item()
        b = inputs["data:geometry:wing:b_50"].item()
        S_w = inputs["data:geometry:wing:area"].item()

        y = np.linspace(0, b / 2, 100)
        L_y = 1 - (y / (b / 2)) ** 2
        ch_vec = np.linspace(c_root, c_tip, 100)
        product = L_y * ch_vec
        integral_result = np.trapz(product, y)
        CL_ref = integral_result / (S_w / 2)

        outputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = 1 / (
            3.14
            * inputs["data:geometry:horizontal_tail:aspect_ratio"]
            * inputs["data:aerodynamics:aircraft:cruise:oswald_coefficient"]
        )

        outputs["data:aerodynamics:wing:low_speed:CL_ref"] = CL_ref
