# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAFuelTankProdWeightPerFU(om.ExplicitComponent):
    """
    Computation of the weight per functional unit. Needed for even though the impact of the tank
    won't be considered
    """

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
            name="data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass",
            units="kg",
            val=np.nan,
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the generator required for a functional unit",
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]

        outputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass_per_fu",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass"]
