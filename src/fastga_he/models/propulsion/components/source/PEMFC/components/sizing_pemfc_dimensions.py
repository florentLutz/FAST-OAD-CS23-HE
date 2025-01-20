# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import logging
from ..constants import POSSIBLE_POSITION

UNDERBELLY_RATIO = 0.8  # Ratio between underbelly width and fuselage width
DIMENSION_RATIO = 0.3528  # Ratio between the effective area length size and the actual FC size
LENGTH_LAYER_RATIO = 3.428e-3  # in meters
_LOGGER = logging.getLogger(__name__)


class SizingPEMFCDimensions(om.ExplicitComponent):
    """
    Computation of the different dimensions of the PEMFC, it will heavily depend on the
    position of the pemfc. If the batteries are in the rear, front or in pods,
    we will use ratios. If the batteries are in the underbelly/wing, we will use fuselage/wing
    dimensions.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc pack",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="underbelly",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the pemfc, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length",
            units="m",
            val=2.0,
            desc="Length of the pemfc, as in the size of the pemfc along the X-axis",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
            units="m",
            val=1.8,
            desc="Width of the pemfc, as in the size of the pemfc along the Y-axis",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height",
            units="m",
            val=1.5,
            desc="Height of the pemfc, as in the size of the pemfc along the Z-axis",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            units="m**2",
            val=np.nan,
            desc="Effective fuel cell area in the stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Number of layer in 1 PEMFC stack",
        )

        if position == "underbelly":
            self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        pemfc_area = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
            / DIMENSION_RATIO**2
        )

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
        ] = (
            LENGTH_LAYER_RATIO
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
        )

        if position in "underbelly":
            _LOGGER.warning(
                msg="Position Underbelly, Fuel cell height and width adjusted for better fitting !!"
            )

            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ] = UNDERBELLY_RATIO * inputs["data:geometry:fuselage:maximum_width"]

            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ] = pemfc_area / (UNDERBELLY_RATIO * inputs["data:geometry:fuselage:maximum_width"])

        else:
            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(pemfc_area)

            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(pemfc_area)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        pemfc_area = (
            inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
            / DIMENSION_RATIO**2
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = LENGTH_LAYER_RATIO

        if position in "underbelly":
            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:geometry:fuselage:maximum_width",
            ] = UNDERBELLY_RATIO

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 1 / (
                DIMENSION_RATIO**2
                * UNDERBELLY_RATIO
                * inputs["data:geometry:fuselage:maximum_width"]
            )

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:geometry:fuselage:maximum_width",
            ] = -pemfc_area / (
                UNDERBELLY_RATIO * inputs["data:geometry:fuselage:maximum_width"] ** 2
            )

        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 / np.sqrt(
                inputs[
                    "data:propulsion:he_power_train:pemfc_stack:"
                    + pemfc_stack_id
                    + ":effective_area"
                ]
                * DIMENSION_RATIO**2
            )

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 / np.sqrt(
                inputs[
                    "data:propulsion:he_power_train:pemfc_stack:"
                    + pemfc_stack_id
                    + ":effective_area"
                ]
                * DIMENSION_RATIO**2
            )
