# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class LCCDCDCConverterCost(om.ExplicitComponent):
    """
    Computation of DC-DC converter cost, regression function, estimated from :cite:`fuhren:2022`.
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
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_max",
            units="V",
            val=np.nan,
        )
        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":cost_per_converter",
            val=1000.0,
            units="USD",
        )

        self.declare_partials(of="*", wrt="*", val=1.45)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":cost_per_converter"
        ] = (
            1.45
            * inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":voltage_out_max"
            ]
            + 652.5
        )
