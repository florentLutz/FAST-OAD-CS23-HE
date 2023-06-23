# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerCleanWingLift(om.ExplicitComponent):
    """
    The computation of the clean wing lift is required since we assume that each section lift
    would scale with it.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(name="alpha", val=np.full(number_of_points, np.nan), units="rad")
        self.add_input(name="data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input(name="data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)

        self.add_output(name="cl_wing_clean", val=0.5, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        alpha = inputs["alpha"]

        cl_wing = cl0_wing + cl_alpha_wing * alpha

        outputs["cl_wing_clean"] = cl_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["cl_wing_clean", "data:aerodynamics:wing:cruise:CL0_clean"] = np.ones(
            number_of_points
        )
        partials["cl_wing_clean", "data:aerodynamics:wing:cruise:CL_alpha"] = inputs["alpha"]
        partials["cl_wing_clean", "alpha"] = (
            np.eye(number_of_points) * inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        )
