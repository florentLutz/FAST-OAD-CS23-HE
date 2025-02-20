# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION

AREA_RATIO = 0.12446784  # Ratio between the effective area length size and the actual FC size
LENGTH_LAYER_RATIO = 3.428e-3  # [m]
DEFAULT_FC_POWER_DENSITY = 124  # [kW/m^3]


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

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="m**2",
            val=np.nan,
            desc="Effective fuel cell area in the stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Number of layer in 1 PEMFC stack",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            units="kW/m**3",
            val=np.nan,
        )

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        effective_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
        ]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
        ]
        power_density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
        ]

        pemfc_area = effective_area / AREA_RATIO
        adjust_factor = DEFAULT_FC_POWER_DENSITY / power_density

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:length"
        ] = LENGTH_LAYER_RATIO * number_of_layers

        if position in "underbelly":
            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(2 * pemfc_area * adjust_factor)

            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(pemfc_area * adjust_factor / 2)

        else:
            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(pemfc_area * adjust_factor)

            outputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(pemfc_area * adjust_factor)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        effective_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
        ]

        power_density = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density"
        ]

        pemfc_area = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
            / AREA_RATIO
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:length",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = LENGTH_LAYER_RATIO

        adjust_factor = DEFAULT_FC_POWER_DENSITY / power_density

        if position in "underbelly":
            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(2 * adjust_factor) / np.sqrt(effective_area * AREA_RATIO)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(adjust_factor) / np.sqrt(2 * effective_area * AREA_RATIO)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            ] = -0.5 * np.sqrt(2 * pemfc_area * DEFAULT_FC_POWER_DENSITY) / power_density**1.5

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            ] = -0.5 * np.sqrt(0.5 * pemfc_area * DEFAULT_FC_POWER_DENSITY) / power_density**1.5

        else:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(adjust_factor) / np.sqrt(effective_area * AREA_RATIO)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(adjust_factor) / np.sqrt(effective_area * AREA_RATIO)

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            ] = -0.5 * np.sqrt(pemfc_area * DEFAULT_FC_POWER_DENSITY) / power_density**1.5

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_density",
            ] = -0.5 * np.sqrt(pemfc_area * DEFAULT_FC_POWER_DENSITY) / power_density**1.5
