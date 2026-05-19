# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCDCConverterOperationalCost(om.ExplicitComponent):
    """
    Computation of convertor annual operational cost. This is estimated based on the lifespan of
    the IGBTs given by :cite:`sathik:2018`, which is indicated as the throttling component.
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
            + ":purchase_cost",
            val=np.nan,
            units="USD",
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":lifespan",
            units="h",
            val=4.2e4,
            desc="Expected lifetime of the DC_DC_converter, based on the lifespan of the IGBTs",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost",
            units="USD/yr",
            val=31.02,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        purchase_cost = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost"
        ]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan"
        ]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost"
        ] = purchase_cost * flight_hours_per_year / lifespan

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        purchase_cost = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost"
        ]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan"
        ]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":purchase_cost",
        ] = flight_hours_per_year / lifespan

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = purchase_cost / lifespan

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":operational_cost",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan",
        ] = -purchase_cost * flight_hours_per_year / lifespan**2.0
