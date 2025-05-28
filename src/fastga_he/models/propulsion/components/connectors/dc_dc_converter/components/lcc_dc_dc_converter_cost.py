# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCDCConverterCost(om.ExplicitComponent):
    """
    Computation of convertor purchase cost. Reference prices obtained from
    https://www.mcico.com/truebluepower/converters?purchase_type=New+Outright%2CNew+Exchange.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dc_power_in_rating",
            val=np.nan,
            units="kW",
            desc="Maximum power rating of the converter",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost",
            units="USD",
            val=750.0,
            desc="Unit purchase cost of the converter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost"
        ] = (
            733.0
            * np.log(
                inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:"
                    + dc_dc_converter_id
                    + ":dc_power_in_rating"
                ]
            )
            + 2295.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":dc_power_in_rating",
        ] = (
            733.0
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":dc_power_in_rating"
            ]
        )
