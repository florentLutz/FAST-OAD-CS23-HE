# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCADCBusProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )

    def setup(self):
        dc_bus_id = self.options["dc_bus_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Weight of the bus bar",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the bus bar required for a functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_bus_id = self.options["dc_bus_id"]

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_bus_id = self.options["dc_bus_id"]

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass_per_fu",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass",
        ] = 1.0 * inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = 1.0 * inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":mass"]
