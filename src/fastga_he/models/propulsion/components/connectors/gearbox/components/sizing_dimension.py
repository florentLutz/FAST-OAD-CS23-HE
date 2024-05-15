# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingGearboxDimensions(om.ExplicitComponent):
    """
    Computation of the dimensions of the gearbox, is based on the scaling factor and a
    reference gearbox. Reference gearbox is single stage planetary gearbox presented here:
    https://www.ato.com/inline-planetary-speed-reducer-gearbox
    """

    def initialize(self):
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )
        self.options.declare(
            "width_ref",
            default=150.0,
            desc="Width of the reference gearbox [mm]",
        )
        self.options.declare(
            "height_ref",
            default=150.0,
            desc="Height of the reference gearbox [mm]",
        )
        self.options.declare(
            "length_ref",
            default=265.0,
            desc="Length of the reference gearbox [mm]",
        )

    def setup(self):

        gearbox_id = self.options["gearbox_id"]

        self.add_input(
            name="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":width",
            val=self.options["width_ref"],
            units="mm",
        )
        self.add_output(
            name="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":height",
            val=self.options["height_ref"],
            units="mm",
        )
        self.add_output(
            name="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":length",
            val=self.options["length_ref"],
            units="mm",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        gearbox_id = self.options["gearbox_id"]

        scaling_factor = inputs[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions"
        ]

        outputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":width"] = (
            self.options["width_ref"] * scaling_factor
        )
        outputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":height"] = (
            self.options["height_ref"] * scaling_factor
        )
        outputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":length"] = (
            self.options["length_ref"] * scaling_factor
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        gearbox_id = self.options["gearbox_id"]

        partials[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":width",
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions",
        ] = self.options["width_ref"]
        partials[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":height",
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions",
        ] = self.options["height_ref"]
        partials[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":length",
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions",
        ] = self.options["length_ref"]
