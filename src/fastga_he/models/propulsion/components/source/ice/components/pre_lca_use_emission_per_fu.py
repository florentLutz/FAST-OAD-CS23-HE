# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

SPECIES_LIST = ["CO2", "CO", "NOx", "SOx", "HC", "H2O", "lead"]


class PreLCAICEUseEmissionPerFU(om.ExplicitComponent):
    """
    Simply dividing the emission to get how much per functional unit we need. Would have
    preferred to separate them all but for the sake of refactoring it's all in one.

    We'll also include the emissions during the line testing phase and possible during distribution
    phase in that component because the computation is very similar. Again I'd have preferred to do
    it separately but for the sake of refactoring it's easier to do it here.
    """

    def initialize(self):
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):
        ice_id = self.options["ice_id"]

        self.add_input(name="data:environmental_impact:flight_per_fu", val=np.nan)

        # Already includes the per FU division
        self.add_input(name="data:environmental_impact:aircraft_per_fu", val=np.nan)
        self.add_input(name="data:environmental_impact:line_test:mission_ratio", val=np.nan)
        self.add_input(name="data:environmental_impact:delivery:mission_ratio", val=np.nan)

        for specie in SPECIES_LIST:
            input_name = "data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie
            self.add_input(name=input_name, val=np.nan, units="kg")

            operation_output_name = (
                "data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )
            self.add_output(name=operation_output_name, val=0.0, units="kg")
            self.declare_partials(
                of=operation_output_name,
                wrt=[input_name, "data:environmental_impact:flight_per_fu"],
            )

            manufacturing_output_name = (
                "data:LCA:manufacturing:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )
            self.add_output(name=manufacturing_output_name, val=0.0, units="kg")
            self.declare_partials(
                of=manufacturing_output_name,
                wrt=[
                    input_name,
                    "data:environmental_impact:aircraft_per_fu",
                    "data:environmental_impact:line_test:mission_ratio",
                ],
            )

            distribution_output_name = (
                "data:LCA:distribution:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )
            self.add_output(name=distribution_output_name, val=0.0, units="kg")
            self.declare_partials(
                of=distribution_output_name,
                wrt=[
                    input_name,
                    "data:environmental_impact:aircraft_per_fu",
                    "data:environmental_impact:delivery:mission_ratio",
                ],
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ice_id = self.options["ice_id"]

        for specie in SPECIES_LIST:
            input_name = "data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie
            operation_output_name = (
                "data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )
            outputs[operation_output_name] = (
                inputs[input_name] * inputs["data:environmental_impact:flight_per_fu"]
            )

            manufacturing_output_name = (
                "data:LCA:manufacturing:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )
            outputs[manufacturing_output_name] = (
                inputs["data:environmental_impact:line_test:mission_ratio"]
                * inputs[input_name]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )

            distribution_output_name = (
                "data:LCA:distribution:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )
            outputs[distribution_output_name] = (
                inputs["data:environmental_impact:delivery:mission_ratio"]
                * inputs[input_name]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ice_id = self.options["ice_id"]

        for specie in SPECIES_LIST:
            input_name = "data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie
            operation_output_name = (
                "data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )

            partials[operation_output_name, input_name] = inputs[
                "data:environmental_impact:flight_per_fu"
            ]
            partials[operation_output_name, "data:environmental_impact:flight_per_fu"] = inputs[
                input_name
            ]

            manufacturing_output_name = (
                "data:LCA:manufacturing:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )

            partials[manufacturing_output_name, input_name] = (
                inputs["data:environmental_impact:line_test:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )
            partials[manufacturing_output_name, "data:environmental_impact:aircraft_per_fu"] = (
                inputs["data:environmental_impact:line_test:mission_ratio"] * inputs[input_name]
            )
            partials[
                manufacturing_output_name, "data:environmental_impact:line_test:mission_ratio"
            ] = inputs["data:environmental_impact:aircraft_per_fu"] * inputs[input_name]

            distribution_output_name = (
                "data:LCA:distribution:he_power_train:ICE:" + ice_id + ":" + specie + "_per_fu"
            )

            partials[distribution_output_name, input_name] = (
                inputs["data:environmental_impact:delivery:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )
            partials[distribution_output_name, "data:environmental_impact:aircraft_per_fu"] = (
                inputs["data:environmental_impact:delivery:mission_ratio"] * inputs[input_name]
            )
            partials[
                distribution_output_name, "data:environmental_impact:delivery:mission_ratio"
            ] = inputs["data:environmental_impact:aircraft_per_fu"] * inputs[input_name]
