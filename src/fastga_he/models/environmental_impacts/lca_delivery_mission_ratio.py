# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCARatioDeliveryFlightMission(om.ExplicitComponent):
    """
    Compute the ratio of fuel needed for the delivery (when it is done by planes) wrt to the fuel
    needed for the mission. This will serve to compute the impact of manufacturing we'll assume
    the emission profiles are similar.
    """

    def setup(self):
        self.add_input("data:TLAR:range", units="km", val=np.nan)
        self.add_input(
            "data:environmental_impact:delivery:distance",
            units="km",
            val=np.nan,
            desc="Distance between the site of assembly of the aircraft and the site of delivery. "
            "The variable will have the same name regardless of whether the delivery is done "
            "by the air or by land.",
        )

        self.add_output(
            "data:environmental_impact:delivery:mission_ratio",
            val=5.0,
            desc="Ratio of fuel used during the delivery to fuel used during mission. Will only "
            "be used if the aircraft is delivered via the air.",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:environmental_impact:delivery:mission_ratio"] = (
            inputs["data:environmental_impact:delivery:distance"] / inputs["data:TLAR:range"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:environmental_impact:delivery:mission_ratio",
            "data:environmental_impact:delivery:distance",
        ] = 1.0 / inputs["data:TLAR:range"]
        partials[
            "data:environmental_impact:delivery:mission_ratio",
            "data:TLAR:range",
        ] = (
            -inputs["data:environmental_impact:delivery:distance"]
            / inputs["data:TLAR:range"] ** 2.0
        )
