# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from openmdao import api as om

import fastoad.api as oad

from fastga.models.weight.mass_breakdown.constants import SUBMODEL_PAYLOAD_MASS

# Register FAST-GA submodel as defaults
oad.RegisterSubmodel.active_models[
    SUBMODEL_PAYLOAD_MASS
] = "fastga.submodel.weight.mass.payload.legacy"


@oad.RegisterSubmodel(SUBMODEL_PAYLOAD_MASS, "fastga_he.weight.payload_for_retrofit")
class ComputePayloadForRetrofit(om.ExplicitComponent):
    """
    In the case of a retrofit where we aim to keep the MTOW constant  we must sacrifice payload
    in order to fit new power-train and their fuel consumption. This component computes the
    payload we can still carry
    """

    def setup(self):

        self.add_input("data:weight:aircraft:target_MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:payload", units="kg", val=500.0)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:weight:aircraft:payload"] = min(
            inputs["data:weight:aircraft:target_MTOW"]
            - inputs["data:weight:aircraft:OWE"]
            - inputs["data:mission:sizing:fuel"],
            inputs["data:weight:aircraft:max_payload"],
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        if (
            inputs["data:weight:aircraft:target_MTOW"]
            - inputs["data:weight:aircraft:OWE"]
            - inputs["data:mission:sizing:fuel"]
            < inputs["data:weight:aircraft:max_payload"]
        ):
            partials["data:weight:aircraft:payload", "data:weight:aircraft:target_MTOW"] = 1.0
            partials["data:weight:aircraft:payload", "data:weight:aircraft:OWE"] = -1.0
            partials["data:weight:aircraft:payload", "data:mission:sizing:fuel"] = -1.0
            partials["data:weight:aircraft:payload", "data:weight:aircraft:max_payload"] = 0.0
        else:
            partials["data:weight:aircraft:payload", "data:weight:aircraft:target_MTOW"] = 0.0
            partials["data:weight:aircraft:payload", "data:weight:aircraft:OWE"] = 0.0
            partials["data:weight:aircraft:payload", "data:mission:sizing:fuel"] = 0.0
            partials["data:weight:aircraft:payload", "data:weight:aircraft:max_payload"] = 1.0
