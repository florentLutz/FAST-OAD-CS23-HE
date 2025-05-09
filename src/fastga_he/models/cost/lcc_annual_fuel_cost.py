# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
import logging

import numpy as np
import openmdao.api as om

_LOGGER = logging.getLogger(__name__)


class LCCAnnualFuelCost(om.ExplicitComponent):
    """
    Computation of the yearly fuel cost of the aircraft. The cost of unit hydrogen is obtained from
    :cite:`sens:2024`. The unit price of avgas 100LL and Jet-A1 are obtained from
    https://orleans.aeroport.fr.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.price_fuel = None

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )

        self.add_output(
            name="data:operation:annual_fuel_cost",
            val=1000.0,
            units="USD/yr",
        )

        for tank_type, tank_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if (comp_type == "fuel_tank" or comp_type == "gaseous_hydrogen_tank")
        ]:
            if tank_type == "fuel_tank":
                self.add_input(
                    "data:propulsion:he_power_train:fuel_tank:" + tank_id + ":fuel_type",
                    val=1.0,
                    desc="Type of fuel stored in the tank, 1.0 - gasoline, 2.0 - Diesel, 3.0 - Jet A1",
                )

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
                wrt=[
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_mission",
                    "data:TLAR:flight_per_year",
                ],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]
        outputs["data:operation:annual_fuel_cost"] = 0.0

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
                    self.price_fuel = 3.66  # gasoline price [USD/kg], Avgas
                elif fuel_type == 2.0:
                    self.price_fuel = 1.977  # Diesel price [USD/kg]
                elif fuel_type == 3.0:
                    self.price_fuel = 2.967  # Jet-A1 price [USD/kg]
                else:
                    self.price_fuel = 3.66
                    _LOGGER.warning("Fuel type %f does not exist, replaced by type 1!", fuel_type)

                outputs["data:operation:annual_fuel_cost"] += (
                    self.price_fuel
                    * flight_per_year
                    * inputs[
                        "data:propulsion:he_power_train:fuel_tank:"
                        + tank_id
                        + ":fuel_consumed_mission"
                    ]
                )

            elif tank_type == "gaseous_hydrogen_tank":
                outputs["data:operation:annual_fuel_cost"] += (
                    6.54
                    * inputs[
                        "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                        + tank_id
                        + ":fuel_consumed_mission"
                    ]
                    * inputs["data:TLAR:flight_per_year"]
                )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]

        for tank_type, tank_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if (comp_type == "fuel_tank" or comp_type == "gaseous_hydrogen_tank")
        ]:
            fuel_consumed = inputs[
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_consumed_mission"
            ]
            if tank_type == "fuel_tank":
                partials[
                    "data:operation:annual_fuel_cost",
                    "data:propulsion:he_power_train:fuel_tank:"
                    + tank_id
                    + ":fuel_consumed_mission",
                ] = self.price_fuel * flight_per_year

                partials["data:operation:annual_fuel_cost", "data:TLAR:flight_per_year"] = (
                    self.price_fuel * fuel_consumed
                )

            elif tank_type == "gaseous_hydrogen_tank":
                partials[
                    "data:operation:annual_fuel_cost",
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + tank_id
                    + ":fuel_consumed_mission",
                ] = 6.54 * flight_per_year

                partials["data:operation:annual_fuel_cost", "data:TLAR:flight_per_year"] = (
                    6.54 * fuel_consumed
                )
