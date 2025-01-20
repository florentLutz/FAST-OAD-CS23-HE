# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPEMFCVolume(om.ExplicitComponent):
    """
    Computation of the volume the PEMFC based on number of layers.
    """

    # TODO: Adding another way of mass estimation from D.Juschus

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
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length",
            units="m",
            val=np.nan,
            desc="Length of the pemfc, as in the size of the pemfc along the X-axis",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
            units="m",
            val=np.nan,
            desc="Width of the pemfc, as in the size of the pemfc along the Y-axis",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height",
            units="m",
            val=np.nan,
            desc="Height of the pemfc, as in the size of the pemfc along the Z-axis",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume",
            units="m**3",
            val=5.0,
            desc="Volume of the pemfc stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume"] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length",
        ] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height",
        ] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":volume",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
        ] = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ]
        )
