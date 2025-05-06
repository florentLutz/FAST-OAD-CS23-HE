# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)


class LCCFuelTankOperationalCost(om.ExplicitComponent):
    """
    Computation of the annual fuel cost. The unit price of avgas 100LL and Jet-A1 are obtained from
    https://orleans.aeroport.fr.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.price_fuel = None

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]

        self.add_input(
            name="data:cost:operation:mission_per_year",
            val=np.nan,
            units="1/yr",
            desc="Flight mission per year",
        )

        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission",
            units="kg",
            val=np.nan,
            desc="Amount of fuel from that tank which will be consumed during mission",
        )

        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_type",
            val=1.0,
            desc="Type of fuel stored in the tank, 1.0 - gasoline, 2.0 - Diesel, 3.0 - Jet A1",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":operational_cost",
            units="USD/yr",
            val=1.0e4,
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":fuel_consumed_mission",
                "data:cost:operation:mission_per_year",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]

        fuel_type = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_type"
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

        outputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":operational_cost"
        ] = (
            self.price_fuel
            * inputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":fuel_consumed_mission"
            ]
            * inputs["data:cost:operation:mission_per_year"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]

        mission_per_year = inputs["data:cost:operation:mission_per_year"]
        fuel_consumed = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission"
        ]

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":operational_cost",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission",
        ] = self.price_fuel * mission_per_year

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":operational_cost",
            "data:cost:operation:mission_per_year",
        ] = self.price_fuel * fuel_consumed
