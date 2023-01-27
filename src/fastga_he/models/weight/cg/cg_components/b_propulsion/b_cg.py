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
    The propulsion CG is computed during the sizing, this is just the interfacing with the OAD
    process.
    """

    def setup(self):

        self.add_input("data:propulsion:he_power_train:CG:x", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:CG:x", units="m", val=2.5)

        self.declare_partials(
            of="data:weight:propulsion:CG:x", wrt="data:propulsion:he_power_train:CG:x", val=1.0
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:weight:propulsion:CG:x"] = inputs["data:propulsion:he_power_train:CG:x"]
