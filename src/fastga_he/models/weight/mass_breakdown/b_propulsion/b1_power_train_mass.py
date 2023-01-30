# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga.models.weight.mass_breakdown.constants import SUBMODEL_PROPULSION_MASS


@oad.RegisterSubmodel(
    SUBMODEL_PROPULSION_MASS, "fastga_he.submodel.weight.mass.propulsion.power_train"
)
class PowerTrainMass(om.ExplicitComponent):
    def initialize(self):
        # Needed even if it isn't used because the original one has that option ...
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        self.add_input("data:propulsion:he_power_train:mass", val=np.nan, units="kg")

        self.add_output("data:weight:propulsion:mass", val=350.0, units="kg")

        self.declare_partials(
            of="data:weight:propulsion:mass", wrt="data:propulsion:he_power_train:mass", val=1.0
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:weight:propulsion:mass"] = inputs["data:propulsion:he_power_train:mass"]
