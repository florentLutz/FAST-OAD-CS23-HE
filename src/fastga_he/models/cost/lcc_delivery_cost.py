# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging
import numpy as np
import openmdao.api as om

_LOGGER = logging.getLogger(__name__)


class LCCDeliveryCost(om.ExplicitComponent):
    """
    Computation of the delivery cost per aircraft obtained from :cite:`stefana:2024`. The train
    freight is referenced from
    https://www.tj-chinafreight.com/why-choose-rail-freight-for-shipping-from-china-to-europe-central-aisa-russia/.
    """

    def initialize(self):
        self.options.declare(
            name="delivery_method",
            default="flight",
            desc="Method with which the aircraft will be brought from the assembly plant to the "
            "end user. Can be either flown or carried by train",
            allow_none=False,
            values=["flight", "train"],
        )

    def setup(self):
        delivery_method = self.options["delivery_method"]

        self.add_output(
            "data:cost:delivery_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Freight cost per aircraft",
        )

        if delivery_method == "train":
            self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")

            self.declare_partials(
                of="data:cost:delivery_cost_per_unit", wrt="data:weight:aircraft:OWE", val=1.3625
            )

        elif delivery_method == "flight":
            self.add_input(
                name="data:cost:electricity_cost",
                val=0.0,
                units="USD",
                desc="Electric energy cost for single flight mission",
            )
            self.add_input(
                name="data:cost:fuel_cost",
                val=0.0,
                units="USD",
                desc="Fuel cost for single flight mission",
            )
            self.add_input(
                name="data:cost:delivery:mission_ratio",
                val=np.nan,
                desc="Ratio between the delivery flight duration and the sizing mission duration",
            )
            self.add_input(
                "data:cost:production:flight_cost_factor",
                val=1.0,
                desc="Adjustment factor to consider other flight delivery related costs",
            )

            self.declare_partials(of="data:cost:delivery_cost_per_unit", wrt="*", method="exact")

        else:
            _LOGGER.warning(
                "Delivery method "
                + delivery_method
                + " does not exist, replaced with a delivery by flight."
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        delivery_method = self.options["delivery_method"]

        if delivery_method == "train":
            outputs["data:cost:delivery_cost_per_unit"] = (
                1.3625 * inputs["data:weight:aircraft:OWE"]
            )

        elif delivery_method == "flight":
            outputs["data:cost:delivery_cost_per_unit"] = (
                (inputs["data:cost:fuel_cost"] + inputs["data:cost:electricity_cost"])
                * inputs["data:cost:production:flight_cost_factor"]
                * inputs["data:cost:delivery:mission_ratio"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        delivery_method = self.options["delivery_method"]

        if delivery_method == "flight":
            cost_factor = inputs["data:cost:production:flight_cost_factor"]
            duration_ratio = inputs["data:cost:delivery:mission_ratio"]
            fuel_cost = inputs["data:cost:fuel_cost"]
            electricity_cost = inputs["data:cost:electricity_cost"]

            partials["data:cost:delivery_cost_per_unit", "data:cost:fuel_cost"] = (
                cost_factor * duration_ratio
            )

            partials["data:cost:delivery_cost_per_unit", "data:cost:electricity_cost"] = (
                cost_factor * duration_ratio
            )

            partials[
                "data:cost:delivery_cost_per_unit", "data:cost:production:flight_cost_factor"
            ] = (fuel_cost + electricity_cost) * duration_ratio

            partials["data:cost:delivery_cost_per_unit", "data:cost:delivery:mission_ratio"] = (
                fuel_cost + electricity_cost
            ) * cost_factor
