# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCARatioDeliveryFlightMission(om.ExplicitComponent):
    """
    Compute the ratio of fuel needed for the delivery (when it is done by planes) wrt to the fuel
    needed for the mission. This will serve to compute the impact of manufacturing we'll assume
    the emission profiles are similar.
    """

    def initialize(self):
        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        if not self.options["use_operational_mission"]:
            self.add_input("data:TLAR:range", units="km", val=np.nan)
        else:
            self.add_input("data:mission:operational:range", units="km", val=np.nan)

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
        if not self.options["use_operational_mission"]:
            range_mission = inputs["data:TLAR:range"]
        else:
            range_mission = inputs["data:mission:operational:range"]

        outputs["data:environmental_impact:delivery:mission_ratio"] = (
            inputs["data:environmental_impact:delivery:distance"] / range_mission
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            range_mission_name = "data:TLAR:range"

        else:
            range_mission_name = "data:mission:operational:range"

        partials[
            "data:environmental_impact:delivery:mission_ratio",
            "data:environmental_impact:delivery:distance",
        ] = 1.0 / inputs[range_mission_name]
        partials[
            "data:environmental_impact:delivery:mission_ratio",
            range_mission_name,
        ] = (
            -inputs["data:environmental_impact:delivery:distance"]
            / inputs[range_mission_name] ** 2.0
        )
