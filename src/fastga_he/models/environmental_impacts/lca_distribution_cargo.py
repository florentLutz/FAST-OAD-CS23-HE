# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCADistributionCargoMassDistancePerFU(om.ExplicitComponent):
    """
    Compute the product mass*delivery distance per functional unit for the case in which the
    aircraft is delivered via train.
    """

    def setup(self):
        self.add_input(
            "data:environmental_impact:delivery:distance",
            units="km",
            val=np.nan,
            desc="Distance between the site of assembly of the aircraft and the site of delivery. "
            "The variable will have the same name regardless of whether the delivery is done "
            "by the air or by land.",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")

        self.add_output(
            "data:environmental_impact:delivery:cargo_transport_per_fu",
            val=5.0e-3,
            units="t*km",
            desc="Product of aircraft mass by the distance it needs to be delivered per "
            "functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:environmental_impact:delivery:cargo_transport_per_fu"] = (
            inputs["data:environmental_impact:delivery:distance"]
            * inputs["data:weight:aircraft:OWE"]
            / 1000.0
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:environmental_impact:delivery:cargo_transport_per_fu",
            "data:environmental_impact:delivery:distance",
        ] = (
            inputs["data:weight:aircraft:OWE"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            / 1000.0
        )
        partials[
            "data:environmental_impact:delivery:cargo_transport_per_fu",
            "data:weight:aircraft:OWE",
        ] = (
            inputs["data:environmental_impact:delivery:distance"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            / 1000.0
        )
        partials[
            "data:environmental_impact:delivery:cargo_transport_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:weight:aircraft:OWE"]
            * inputs["data:environmental_impact:delivery:distance"]
            / 1000.0
        )
