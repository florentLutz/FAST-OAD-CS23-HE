# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

SPECIES_LIST = ["CO2", "CO", "NOx", "SOx", "HC", "H2O"]


class PreLCABatteryUseEmissionPerFU(om.ExplicitComponent):
    """
    Simply dividing the emission to get how much per functional unit we need. Would have
    preferred to separate them all but for the sake of refactoring it's all in one.
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(name="data:environmental_impact:flight_per_fu", val=1e-3)

        for specie in SPECIES_LIST:
            input_name = (
                "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":" + specie
            )
            self.add_input(name=input_name, val=np.nan, units="kg")

            output_name = (
                "data:LCA:operation:he_power_train:battery_pack:"
                + battery_pack_id
                + ":"
                + specie
                + "_per_fu"
            )
            self.add_output(name=output_name, val=0.0, units="kg")

            self.declare_partials(
                of=output_name, wrt=[input_name, "data:environmental_impact:flight_per_fu"]
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        for specie in SPECIES_LIST:
            input_name = (
                "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":" + specie
            )
            output_name = (
                "data:LCA:operation:he_power_train:battery_pack:"
                + battery_pack_id
                + ":"
                + specie
                + "_per_fu"
            )

            outputs[output_name] = (
                inputs[input_name] * inputs["data:environmental_impact:flight_per_fu"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        for specie in SPECIES_LIST:
            input_name = (
                "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":" + specie
            )
            output_name = (
                "data:LCA:operation:he_power_train:battery_pack:"
                + battery_pack_id
                + ":"
                + specie
                + "_per_fu"
            )

            partials[output_name, input_name] = inputs["data:environmental_impact:flight_per_fu"]
            partials[output_name, "data:environmental_impact:flight_per_fu"] = inputs[input_name]
