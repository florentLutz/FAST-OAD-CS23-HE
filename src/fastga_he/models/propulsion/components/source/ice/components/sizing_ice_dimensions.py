# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingICEDimensions(om.ExplicitComponent):
    """
    Computation of the dimensions of the ICE. Based on the scaling factor and the dimensions of
    the reference ICE, the Lycoming IO-360-B1A.
    """

    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )
        self.options.declare(
            "length_ref",
            default=0.83,
            desc="Length of the reference motor in [m]",
        )
        self.options.declare(
            "width_ref",
            default=0.85,
            desc="Width of the reference motor in [m]",
        )
        self.options.declare(
            "height_ref",
            default=0.57,
            desc="Height of the reference motor in [m]",
        )

    def setup(self):

        ice_id = self.options["ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:length",
            val=np.nan,
            desc="Scaling factor for the length of the ICE",
        )
        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:width",
            val=np.nan,
            desc="Scaling factor for the width of the ICE",
        )
        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:height",
            val=np.nan,
            desc="Scaling factor for the height of the ICE",
        )

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length",
            val=self.options["length_ref"],
            desc="Length of the ICE",
            units="m",
        )
        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:width",
            val=self.options["width_ref"],
            desc="Width of the ICE",
            units="m",
        )
        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:height",
            val=self.options["height_ref"],
            desc="Height of the ICE",
            units="m",
        )

        self.declare_partials(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:length",
            val=self.options["length_ref"],
        )
        self.declare_partials(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:width",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:width",
            val=self.options["width_ref"],
        )
        self.declare_partials(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":engine:height",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:height",
            val=self.options["height_ref"],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":engine:length"] = (
            inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:length"]
            * self.options["length_ref"]
        )
        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":engine:width"] = (
            inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:width"]
            * self.options["width_ref"]
        )
        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":engine:height"] = (
            inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":scaling:height"]
            * self.options["height_ref"]
        )
