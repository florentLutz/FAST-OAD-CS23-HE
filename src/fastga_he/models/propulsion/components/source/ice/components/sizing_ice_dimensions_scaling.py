# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import openmdao.api as om
import numpy as np


class SizingICEDimensionsScaling(om.ExplicitComponent):
    """
    Computation of the scaling factor for the dimensions of the ICE. It assumes a constant power
    density and that the ratios between dimensions is constant. The reference engine is a
    Lycoming IO-360-B1A.
    """

    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )
        self.options.declare(
            "power_rating_SL_ref",
            default=132.480,
            desc="Max power at sea level of the reference motor in [kW]",
        )

    def setup(self):

        ice_id = self.options["ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            units="kW",
            val=np.nan,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:length",
            val=1.0,
            desc="Scaling factor for the length of the ICE",
        )
        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:width",
            val=1.0,
            desc="Scaling factor for the width of the ICE",
        )
        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:height",
            val=1.0,
            desc="Scaling factor for the height of the ICE",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        power_rating_sl = inputs[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"
        ]
        power_rating_sl_ref = self.options["power_rating_SL_ref"]

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:length"] = (
            power_rating_sl / power_rating_sl_ref
        ) ** (1.0 / 3.0)
        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:width"] = (
            power_rating_sl / power_rating_sl_ref
        ) ** (1.0 / 3.0)
        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:height"] = (
            power_rating_sl / power_rating_sl_ref
        ) ** (1.0 / 3.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ice_id = self.options["ice_id"]

        power_rating_sl = inputs[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"
        ]
        power_rating_sl_ref = self.options["power_rating_SL_ref"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:length",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
        ] = (
            1.0 / 3.0 * power_rating_sl_ref ** (-1.0 / 3.0) * power_rating_sl ** (-2.0 / 3.0)
        )
        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:width",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
        ] = (
            1.0 / 3.0 * power_rating_sl_ref ** (-1.0 / 3.0) * power_rating_sl ** (-2.0 / 3.0)
        )
        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:height",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
        ] = (
            1.0 / 3.0 * power_rating_sl_ref ** (-1.0 / 3.0) * power_rating_sl ** (-2.0 / 3.0)
        )
