# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAFuelSystemProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="fuel_system_id",
            default=None,
            desc="Identifier of the fuel system",
            types=str,
            allow_none=False,
        )

    def setup(self):
        fuel_system_id = self.options["fuel_system_id"]

        self.add_input(
            name="data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Weight of the fuel system",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the fuel system required for a functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_system_id = self.options["fuel_system_id"]

        outputs["data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_system_id = self.options["fuel_system_id"]

        partials[
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass_per_fu",
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass"]
