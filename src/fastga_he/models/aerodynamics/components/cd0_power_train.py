# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga.models.aerodynamics.constants import SUBMODEL_CD0_NACELLE


@oad.RegisterSubmodel(
    SUBMODEL_CD0_NACELLE, "fastga_he.submodel.aerodynamics.powertrain.cd0.from_pt_file"
)
class Cd0PowerTrain(om.ExplicitComponent):
    """
    This is a component to do the interfacing of the computation of the CD0 in the power train
    with the OAD process. Despite its name inherited from FAST-OAD_CS23 this is indeed the CD0 of
    the power train an not just the nacelle.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        # Again option is required to maintain compatibility with submodels
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:propulsion:he_power_train:" + ls_tag + ":CD0", val=np.nan)
        self.add_output("data:aerodynamics:nacelles:" + ls_tag + ":CD0", val=0.0)
        self.declare_partials(
            of="data:aerodynamics:nacelles:" + ls_tag + ":CD0",
            wrt="data:propulsion:he_power_train:" + ls_tag + ":CD0",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        outputs["data:aerodynamics:nacelles:" + ls_tag + ":CD0"] = inputs[
            "data:propulsion:he_power_train:" + ls_tag + ":CD0"
        ]
