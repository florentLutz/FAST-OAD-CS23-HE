# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)


class LCCFuelTankOperation(om.ExplicitComponent):
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

        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]

        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        self.add_input(
            name=duration_mission_name,
            units="h",
            val=np.nan,
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
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":annual_fuel_cost",
            units="USD/yr",
            val=1.0e4,
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":fuel_consumed_mission",
                duration_mission_name,
                "data:TLAR:flight_hours_per_year",
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

        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        outputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":annual_fuel_cost"
        ] = (
            self.price_fuel
            * inputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":fuel_consumed_mission"
            ]
            * inputs["data:TLAR:flight_hours_per_year"]
            / inputs[duration_mission_name]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]

        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        year_flight_hour = inputs["data:TLAR:flight_hours_per_year"]
        mission_time = inputs[duration_mission_name]
        fuel_consumed = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission"
        ]

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":annual_fuel_cost",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission",
        ] = self.price_fuel * year_flight_hour / mission_time

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":annual_fuel_cost",
            "data:TLAR:flight_hours_per_year",
        ] = self.price_fuel * fuel_consumed / mission_time

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":annual_fuel_cost",
            duration_mission_name,
        ] = -self.price_fuel * fuel_consumed * year_flight_hour / mission_time**2.0
