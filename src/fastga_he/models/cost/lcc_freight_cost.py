# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCFreightCost(om.ExplicitComponent):
    """
    Computation of the delivery cost per aircraft obtained from :cite:`stefana:2024`. The train
    freight is referenced from
    https://www.tj-chinafreight.com/why-choose-rail-freight-for-shipping-from-china-to-europe-central-aisa-russia/.
    """

    def setup(self):
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")

        self.add_input(
            "data:cost:train_delivery",
            val=0.0,
            desc="Portion of aircraft delivered by train",
        )

        self.add_input(
            "data:cost:truck_delivery",
            val=0.0,
            desc="Portion of aircraft delivered by truck",
        )

        self.add_input(
            "data:cost:ship_delivery",
            val=0.0,
            desc="Portion of aircraft delivered by ship",
        )

        self.add_input(
            "data:cost:airplane_delivery",
            val=0.0,
            desc="Portion of aircraft delivered by airplane",
        )

        self.add_output(
            "data:cost:freight_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Freight cost per aircraft",
        )
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:freight_cost_per_unit"] = inputs["data:weight:aircraft:OWE"] * (
            1.09 * inputs["data:cost:ship_delivery"]
            + 5.995 * inputs["data:cost:airplane_delivery"]
            + 1.635 * inputs["data:cost:truck_delivery"]
            + 1.3625 * inputs["data:cost:train_delivery"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        owe = inputs["data:weight:aircraft:OWE"]
        ship = inputs["data:cost:ship_delivery"]
        airplane = inputs["data:cost:airplane_delivery"]
        truck = inputs["data:cost:truck_delivery"]
        train = inputs["data:cost:train_delivery"]

        partials["data:cost:freight_cost_per_unit", "data:weight:aircraft:OWE"] = (
            1.09 * ship + 5.995 * airplane + 1.635 * truck + 1.3625 * train
        )

        partials["data:cost:freight_cost_per_unit", "data:cost:ship_delivery"] = 1.09 * owe
        partials["data:cost:freight_cost_per_unit", "data:cost:airplane_delivery"] = 5.995 * owe
        partials["data:cost:freight_cost_per_unit", "data:cost:truck_delivery"] = 1.635 * owe
        partials["data:cost:freight_cost_per_unit", "data:cost:train_delivery"] = 1.3625 * owe
