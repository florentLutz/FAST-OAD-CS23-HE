# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
from openmdao import api as om

import fastoad.api as oad

from fastga.models.weight.mass_breakdown.constants import SERVICE_PAYLOAD_MASS

# Register FAST-GA submodel as defaults
oad.RegisterSubmodel.active_models[SERVICE_PAYLOAD_MASS] = (
    "fastga.submodel.weight.mass.payload.legacy"
)


@oad.RegisterSubmodel(SERVICE_PAYLOAD_MASS, "fastga_he.weight.payload_for_retrofit")
class ComputePayloadForRetrofit(om.Group):
    """
    This group computes the payload and checks if the maximum payload is higher than the target
    maximum payload. It is used in the case of a retrofit while the MTOW is set to constant values.
    """

    def setup(self):
        self.add_subsystem(
            name="check_max_payload",
            subsys=_CheckMaxPayload(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="compute_payload",
            subsys=_ComputePayload(),
            promotes=["*"],
        )


class _CheckMaxPayload(om.ExplicitComponent):
    """
    This component checks if the calculated maximum payload is higher than the target maximum
    payload.
    """

    def setup(self):
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:target_max_payload", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:max_payload", val=600.0, units="kg")

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        max_payload_ceiling = (
            inputs["data:weight:aircraft:MTOW"] - inputs["data:weight:aircraft:OWE"]
        )
        target_max_payload = inputs["data:weight:aircraft:target_max_payload"]

        outputs["data:weight:aircraft:max_payload"] = np.clip(
            target_max_payload, 0.0, max_payload_ceiling
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        max_payload_ceiling = (
            inputs["data:weight:aircraft:MTOW"] - inputs["data:weight:aircraft:OWE"]
        )
        target_max_payload = inputs["data:weight:aircraft:target_max_payload"]

        partials["data:weight:aircraft:max_payload", "data:weight:aircraft:MTOW"] = np.where(
            target_max_payload >= max_payload_ceiling,
            1.0,
            0.0,
        )
        partials["data:weight:aircraft:max_payload", "data:weight:aircraft:OWE"] = np.where(
            target_max_payload >= max_payload_ceiling,
            -1.0,
            0.0,
        )
        partials["data:weight:aircraft:max_payload", "data:weight:aircraft:target_max_payload"] = (
            np.where(
                (target_max_payload < max_payload_ceiling) & (target_max_payload > 0.0),
                1.0,
                0.0,
            )
        )


class _ComputePayload(om.ExplicitComponent):
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

    def setup_partials(self):
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
