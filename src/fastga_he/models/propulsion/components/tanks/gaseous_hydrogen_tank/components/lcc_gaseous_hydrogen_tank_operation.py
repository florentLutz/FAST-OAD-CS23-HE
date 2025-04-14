# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCGaseousHydrogenTankOperation(om.ExplicitComponent):
    """
    Computation of the hydrogen cost based on estimation provides from:cite:`sens:2024`.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

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
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
            units="kg",
            val=np.nan,
            desc="Amount of fuel from that tank which will be consumed during mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":annual_fuel_cost",
            units="USD/yr",
            val=200.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":annual_fuel_cost"
        ] = (
            6.54
            * inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_consumed_mission"
            ]
            * inputs["data:TLAR:flight_hours_per_year"]
            / inputs[duration_mission_name]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        year_flight_hour = inputs["data:TLAR:flight_hours_per_year"]
        mission_time = inputs[duration_mission_name]
        fuel_consumed = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission"
        ]

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":annual_fuel_cost",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
        ] = 6.54 * year_flight_hour / mission_time

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":annual_fuel_cost",
            "data:TLAR:flight_hours_per_year",
        ] = 6.54 * fuel_consumed / mission_time

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":annual_fuel_cost",
            duration_mission_name,
        ] = -6.54 * fuel_consumed * year_flight_hour / mission_time**2.0
