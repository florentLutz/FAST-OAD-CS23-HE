# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastga.models.weight.cg.cg_components.constants import SUBMODEL_PROPULSION_CG


@oad.RegisterSubmodel(SUBMODEL_PROPULSION_CG, "fastga_he.submodel.weight.cg.propulsion.power_train")
class PowerTrainCG(om.ExplicitComponent):
    """
    For now, the propulsion CG will be simplified a lot so that we can test the process on
    aircraft level. It will be changed in the future to be done in the sizing.
    """

    def setup(self):

        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:CG:x", units="m", val=3.0)

        self.declare_partials(
            of="data:weight:propulsion:CG:x", wrt="data:geometry:wing:MAC:at25percent:x", val=0.5
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:weight:propulsion:CG:x"] = (
            0.5 * inputs["data:geometry:wing:MAC:at25percent:x"]
        )
