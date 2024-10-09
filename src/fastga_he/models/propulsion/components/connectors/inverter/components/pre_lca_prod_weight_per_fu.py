# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAInverterProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):
        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the inverter",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Mass of the inverter required for a functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
        ] = 1.0 * inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = 1.0 * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"]
