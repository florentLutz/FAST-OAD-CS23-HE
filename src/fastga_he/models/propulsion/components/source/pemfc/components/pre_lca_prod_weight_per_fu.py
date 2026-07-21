# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAPEMFCStackProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Installed weight of the PEMFC_stack engine",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":lifespan_flight_hours",
            units="h",
            val=12.5e3,
            desc="The PEMFC stack lifespan in hours",
        )
        self.add_input(
            name="data:TLAR:max_airframe_hours",
            val=3524.9,
            units="h",
            desc="Expected lifetime of the aircraft expressed in airframe hours",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the PEMFC_stack required for a functional unit",
        )

    def setup_partials(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass_per_fu",
            wrt=[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
                "data:environmental_impact:aircraft_per_fu",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass_per_fu",
            wrt=[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":lifespan_flight_hours",
                "data:TLAR:max_airframe_hours",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * np.ceil(
                inputs["data:TLAR:max_airframe_hours"]
                / inputs[
                    "data:propulsion:he_power_train:PEMFC_stack:"
                    + pemfc_stack_id
                    + ":lifespan_flight_hours"
                ]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass_per_fu",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * np.ceil(
            inputs["data:TLAR:max_airframe_hours"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":lifespan_flight_hours"
            ]
        )
        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass"
        ] * np.ceil(
            inputs["data:TLAR:max_airframe_hours"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":lifespan_flight_hours"
            ]
        )
