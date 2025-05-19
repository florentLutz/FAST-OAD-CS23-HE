# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging
import numpy as np
import openmdao.api as om

from .constants import FUEL_STORAGE_TYPES

_LOGGER = logging.getLogger(__name__)


class LCCFuelCost(om.ExplicitComponent):
    """
    Computation of the fuel cost of the aircraft for single mission. The cost of unit hydrogen is
    obtained from :cite:`sens:2024`. The unit price of avgas 100LL and Jet-A1 are obtained from
    https://orleans.aeroport.fr.
    """

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        self.add_output(
            name="data:cost:fuel_cost",
            val=1000.0,
            units="USD",
            desc="Fuel cost for single flight mission",
        )

        for tank_type, tank_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if comp_type in FUEL_STORAGE_TYPES
        ]:
            self.add_input(
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_consumed_mission",
                units="kg",
                val=np.nan,
                desc="Amount of fuel from that tank which will be consumed during mission",
            )

            self.declare_partials(
                of="*",
                wrt="data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_consumed_mission",
                method="exact",
            )

            if tank_type == "fuel_tank":
                self.add_input(
                    "data:propulsion:he_power_train:fuel_tank:" + tank_id + ":fuel_type",
                    val=1.0,
                    desc="Type of fuel stored in the tank, 1.0 - gasoline, 2.0 - Diesel, 3.0 - Jet A1",
                )

                self.declare_partials(
                    "*",
                    "data:propulsion:he_power_train:fuel_tank:" + tank_id + ":fuel_type",
                    method="fd",
                )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]
        outputs["data:cost:fuel_cost"] = 0.0

        for tank_type, tank_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if (comp_type == "fuel_tank" or comp_type == "gaseous_hydrogen_tank")
        ]:
            if tank_type == "fuel_tank":
                fuel_type = inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + tank_id + ":fuel_type"
                ]

                if fuel_type == 1.0:
                    price_fuel = 3.66  # gasoline price [USD/kg], Avgas
                elif fuel_type == 2.0:
                    price_fuel = 1.977  # Diesel price [USD/kg]
                elif fuel_type == 3.0:
                    price_fuel = 2.967  # Jet-A1 price [USD/kg]
                else:
                    price_fuel = 3.66
                    _LOGGER.warning("Fuel type %f does not exist, replaced by type 1!", fuel_type)

                outputs["data:cost:fuel_cost"] += (
                    price_fuel
                    * inputs[
                        "data:propulsion:he_power_train:fuel_tank:"
                        + tank_id
                        + ":fuel_consumed_mission"
                    ]
                )

            elif tank_type == "gaseous_hydrogen_tank":
                outputs["data:cost:fuel_cost"] += (
                    6.54
                    * inputs[
                        "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                        + tank_id
                        + ":fuel_consumed_mission"
                    ]
                )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        for tank_type, tank_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if (comp_type == "fuel_tank" or comp_type == "gaseous_hydrogen_tank")
        ]:
            if tank_type == "fuel_tank":
                fuel_type = inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + tank_id + ":fuel_type"
                ]

                if fuel_type == 1.0:
                    price_fuel = 3.66  # gasoline price [USD/kg], Avgas
                elif fuel_type == 2.0:
                    price_fuel = 1.977  # Diesel price [USD/kg]
                elif fuel_type == 3.0:
                    price_fuel = 2.967  # Jet-A1 price [USD/kg]
                else:
                    price_fuel = 3.66

                partials[
                    "data:cost:fuel_cost",
                    "data:propulsion:he_power_train:fuel_tank:"
                    + tank_id
                    + ":fuel_consumed_mission",
                ] = price_fuel

            elif tank_type == "gaseous_hydrogen_tank":
                partials[
                    "data:cost:fuel_cost",
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + tank_id
                    + ":fuel_consumed_mission",
                ] = 6.54
