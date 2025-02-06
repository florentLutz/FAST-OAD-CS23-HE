# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingGaseousHydrogenTankTotalHydrogenMission(om.ExplicitComponent):
    """
    Computation of the amount of the total amount of hydrogen loaded for the mission. Is the sum of
    the consumed hydrogen and unusable hydrogen.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
            units="kg",
            val=15.0,
            desc="Amount of hydrogen from that tank which will be consumed during mission",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure",
            val=np.nan,
            units="atm",
            desc="gaseous hydrogen tank storage pressure",
        )

        self.add_input(
            name="hydrogen_reactant_pressure",
            units="atm",
            val=0.3,
            desc="hydrogen gas pressure applied at source components",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            units="kg",
            val=15.15,
            desc="Total amount of hydrogen loaded in the tank for the mission",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        storage_pressure = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure"
        ]

        reactant_pressure = inputs["hydrogen_reactant_pressure"]
        mission_consumed_fuel = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission"
        ]
        pressure_ratio = (storage_pressure - reactant_pressure) / storage_pressure
        # pressure ratio between the available pressure and storage pressure
        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission"
        ] = mission_consumed_fuel / pressure_ratio

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        storage_pressure = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure"
        ]

        reactant_pressure = inputs["hydrogen_reactant_pressure"]
        mission_consumed_fuel = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission"
        ]
        pressure_ratio = (storage_pressure - reactant_pressure) / storage_pressure

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
        ] = 1 / pressure_ratio

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":tank_pressure",
        ] = -mission_consumed_fuel * reactant_pressure / (storage_pressure - reactant_pressure) ** 2

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            "hydrogen_reactant_pressure",
        ] = mission_consumed_fuel * storage_pressure / (storage_pressure - reactant_pressure) ** 2
