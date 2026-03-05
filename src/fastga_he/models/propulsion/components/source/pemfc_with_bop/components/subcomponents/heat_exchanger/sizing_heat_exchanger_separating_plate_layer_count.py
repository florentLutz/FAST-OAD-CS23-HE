# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHeatExchangerSeparatingPlateLayerCount(om.ExplicitComponent):
    """
    Number of separating plates and the heat exchange layers in the heat exchanger.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:layer_count",
            desc="Layer count of a single fluid in the exchanger, often referred to the cold "
            "fluid layer count",
            units="unitless",
            val=50.0,
        )

        self.add_output(
            name="air_layer_count",
            units="unitless",
            val=50,
        )
        self.add_output(
            name="coolant_layer_count",
            units="unitless",
            val=49,
        )
        self.add_output(
            name="separating_plate_count",
            units="unitless",
            val=98,
        )

    def setup_partials(self):
        self.declare_partials(["air_layer_count", "coolant_layer_count"], "*", val=1.0)
        self.declare_partials("separating_plate_count", "*", val=2.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        outputs["coolant_layer_count"] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:layer_count"
            ]
            - 1.0
        )
        outputs["air_layer_count"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":heat_exchanger:layer_count"
        ]
        outputs["separating_plate_count"] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":heat_exchanger:layer_count"
            ]
            - 2.0
        )
