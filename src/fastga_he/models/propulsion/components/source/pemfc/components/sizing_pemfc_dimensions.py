# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION

CELL_LENGTH = 3.428e-3  # [m]


class SizingPEMFCStackDimensions(om.ExplicitComponent):
    """
    Computing PEMFC dimensions based on the ratio of its maximum power density to that of the
    Aerostak 200W reference enables more realistic sizing. The calculation is based on the
    equations given by :cite:`hoogendoorn:2018`.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC pack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of PEMFC, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            units="m**3",
            val=np.nan,
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Number of layer in 1 PEMFC stack",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:length",
            units="m",
            val=2.0,
            desc="Length of PEMFC, as in the size of PEMFC along the X-axis",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
            units="m",
            val=1.8,
            desc="Width of PEMFC, as in the size of PEMFC along the Y-axis",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:height",
            units="m",
            val=1.5,
            desc="Height of PEMFC, as in the size of PEMFC along the Z-axis",
        )

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
        ]
        volume = inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:length"
        ] = CELL_LENGTH * number_of_layers

        area = volume / (CELL_LENGTH * number_of_layers)

        if position in "underbelly":
            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(2 * area)

            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(area / 2)

        else:
            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(area)

            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(area)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
        ]
        volume = inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:length",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = CELL_LENGTH

        if position in "underbelly":
            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            ] = 0.5 * np.sqrt(2) / np.sqrt(CELL_LENGTH * number_of_layers * volume)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":number_of_layers",
            ] = -0.5 * np.sqrt(2 * volume) / np.sqrt(CELL_LENGTH * number_of_layers**3)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            ] = 0.5 * np.sqrt(0.5) / np.sqrt(CELL_LENGTH * number_of_layers * volume)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":number_of_layers",
            ] = -0.5 * np.sqrt(0.5 * volume) / np.sqrt(CELL_LENGTH * number_of_layers**3)

        else:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            ] = 0.5 / np.sqrt(CELL_LENGTH * number_of_layers * volume)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":number_of_layers",
            ] = -0.5 * np.sqrt(volume) / np.sqrt(CELL_LENGTH * number_of_layers**3)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":volume",
            ] = 0.5 / np.sqrt(CELL_LENGTH * number_of_layers * volume)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":number_of_layers",
            ] = -0.5 * np.sqrt(volume) / np.sqrt(CELL_LENGTH * number_of_layers**3)
