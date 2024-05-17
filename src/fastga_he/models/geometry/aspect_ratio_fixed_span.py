# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain


@oad.RegisterOpenMDAOSystem(
    "fastga_he.geometry.aspect_ratio_fixed_span", domain=ModelDomain.GEOMETRY
)
class AspectRatioFromTargetSpan(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:wing:target_span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:geometry:wing:aspect_ratio", val=9.0)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:geometry:wing:aspect_ratio"] = (
            inputs["data:geometry:wing:target_span"] ** 2.0 / inputs["data:geometry:wing:area"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials[
            "data:geometry:wing:aspect_ratio", "data:geometry:wing:target_span"
        ] = 2.0 * np.sqrt(
            inputs["data:geometry:wing:area"] / inputs["data:geometry:wing:target_span"]
        )
        partials["data:geometry:wing:aspect_ratio", "data:geometry:wing:area"] = (
            -inputs["data:geometry:wing:target_span"] ** 2.0
            / inputs["data:geometry:wing:area"] ** 2.0
        )
