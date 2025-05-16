# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCDCConverterOperationalCost(om.ExplicitComponent):
    """
    Computation of convertor annual operational cost.
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
            + ":cost_per_unit",
            val=np.nan,
            units="USD",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the DC_DC_converter, typically around 15 year",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost",
            units="USD/yr",
            val=75.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        cost = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":cost_per_unit"
        ]

        lifespan = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan"
        ]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost"
        ] = cost / lifespan

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        cost = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":cost_per_unit"
        ]

        lifespan = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan"
        ]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":cost_per_unit",
        ] = 1.0 / lifespan

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan",
        ] = -cost / lifespan**2.0
