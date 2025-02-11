# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION

DIMENSION_RATIO = 0.3528  # Ratio between the effective area length size and the actual FC size
LENGTH_LAYER_RATIO = 3.428e-3  # in meters
DEFAULT_FC_POWER_DENSITY = 124  # kW/m^3


class SizingPEMFCDimensionsAerostak200W(om.ExplicitComponent):
    """
    Computation of the different dimensions of the PEMFC based on Aerostak 200W.
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

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        effective_area = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
        ]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
        ]

        pemfc_area = effective_area / DIMENSION_RATIO**2

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
        ] = LENGTH_LAYER_RATIO * number_of_layers

        if position in "underbelly":
            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(2 * pemfc_area)

            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(pemfc_area / 2)

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

        effective_area = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
        ]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = LENGTH_LAYER_RATIO

        if position in "underbelly":
            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(2) / np.sqrt(effective_area * DIMENSION_RATIO**2)

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 / np.sqrt(2 * effective_area * DIMENSION_RATIO**2)

        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 / np.sqrt(effective_area * DIMENSION_RATIO**2)

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 / np.sqrt(effective_area * DIMENSION_RATIO**2)


class SizingPEMFCDimensionsPowerDensity(om.ExplicitComponent):
    """
    Computing the PEMFC dimensions based on the ratio of its maximum power density to that
    of the Aerostak 200W reference enables more realistic sizing.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc pack",
            allow_none=False,
        )

        self.options.declare(
            name="underbelly_ratio",
            default=0.8,
            desc="Ratio between underbelly width and fuselage width",
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

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":max_power_density",
            units="kW/m**3",
            val=np.nan,
        )

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        effective_area = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
        ]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
        ]
        max_power_density = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_power_density"
        ]

        pemfc_area = effective_area / DIMENSION_RATIO**2
        adjust_factor = DEFAULT_FC_POWER_DENSITY / max_power_density

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:length"
        ] = LENGTH_LAYER_RATIO * number_of_layers

        if position in "underbelly":
            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(2 * pemfc_area * adjust_factor)

            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(pemfc_area * adjust_factor / 2)

        else:
            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:height"
            ] = np.sqrt(pemfc_area * adjust_factor)

            outputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width"
            ] = np.sqrt(pemfc_area * adjust_factor)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        effective_area = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
        ]

        max_power_density = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_power_density"
        ]

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

        adjust_factor = DEFAULT_FC_POWER_DENSITY / max_power_density

        if position in "underbelly":
            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(2 * adjust_factor) / np.sqrt(effective_area * DIMENSION_RATIO**2)

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(adjust_factor) / np.sqrt(2 * effective_area * DIMENSION_RATIO**2)

            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
            ] = -0.5 * np.sqrt(2 * pemfc_area * DEFAULT_FC_POWER_DENSITY) / max_power_density**1.5

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
            ] = -0.5 * np.sqrt(0.5 * pemfc_area * DEFAULT_FC_POWER_DENSITY) / max_power_density**1.5

        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(adjust_factor) / np.sqrt(effective_area * DIMENSION_RATIO**2)

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            ] = 0.5 * np.sqrt(adjust_factor) / np.sqrt(effective_area * DIMENSION_RATIO**2)

            partials[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":dimension:width",
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
            ] = -0.5 * np.sqrt(pemfc_area * DEFAULT_FC_POWER_DENSITY) / max_power_density**1.5

            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
            ] = -0.5 * np.sqrt(pemfc_area * DEFAULT_FC_POWER_DENSITY) / max_power_density**1.5
