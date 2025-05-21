# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingPEMFCStackDrag(om.ExplicitComponent):
    """
    Computation of the PEMFC stack's contribution to profile drag according to the position given
    in the options. The influence will be calculated if positioned outside the fuselage.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the PEMFC stack, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]

        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        # At least one input is needed regardless of the case
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
            units="m",
            val=np.nan,
            desc="Width of the PEMFC stack, as in the size of the PEMFC stack along the Y-axis",
        )

        if position == "underbelly":
            self.add_input(
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:length",
                units="m",
                val=np.nan,
                desc="Length of the PEMFC stack, as in the size of the PEMFC stack along the "
                "X-axis",
            )

            self.add_input(
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
                units="m",
                val=np.nan,
                desc="Height of the PEMFC stack, as in the size of the PEMFC stack along the Z-axis",
            )

            self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")

            self.add_input("data:aerodynamics:fuselage:" + ls_tag + ":CD0", val=np.nan)

        elif position == "wing_pod":
            self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":" + ls_tag + ":CD0",
            val=0.0,
        )

    def setup_partials(self):
            self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        if position == "wing_pod":
            # According to :cite:`gudmundsson:2013`. the drag of a streamlined external tank,
            # which more or less resemble a podded PEMFC can be computed using the following
            # formula. It highly depends on the tank/wing interface so we will take a middle.
            # Also, there is no dependency on the tank length

            wing_area = inputs["data:geometry:wing:area"]

            frontal_area = (
                np.pi
                * inputs[
                    "data:propulsion:he_power_train:PEMFC_stack:"
                    + pemfc_stack_id
                    + ":dimension:width"
                ]
                ** 2
                / 2.0
            )

            cd0 = 0.10 * frontal_area / wing_area

        elif position == "underbelly":
            cd0_fus = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]

            wet_area = inputs["data:geometry:fuselage:wet_area"]

            belly_width = inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width"
            ]

            belly_length = inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:length"
            ]

            belly_height = inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:height"
            ]

            added_wet_area = (
                belly_length * belly_width
                + 2.0 * belly_length * belly_height
                + 2.0 * belly_height * belly_width
            )

            cd0 = added_wet_area / wet_area * cd0_fus

        else:
            cd0 = 0.0

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":" + ls_tag + ":CD0"
        ] = cd0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        position = self.options["position"]
        low_speed_aero = self.options["low_speed_aero"]
        ls_tag = "low_speed" if low_speed_aero else "cruise"

        if position == "wing_pod":
            frontal_area = (
                np.pi
                * inputs[
                    "data:propulsion:he_power_train:PEMFC_stack:"
                    + pemfc_stack_id
                    + ":dimension:width"
                ]
                ** 2
                / 2.0
            )

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:geometry:wing:area",
            ] = -0.10 * frontal_area / inputs["data:geometry:wing:area"] ** 2.0

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
            ] = (
                0.10
                * np.pi
                * inputs[
                    "data:propulsion:he_power_train:PEMFC_stack:"
                    + pemfc_stack_id
                    + ":dimension:width"
                ]
                / inputs["data:geometry:wing:area"]
            )

        elif position == "underbelly":
            cd0_fus = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]
            wet_area = inputs["data:geometry:fuselage:wet_area"]

            belly_width = inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width"
            ]

            belly_length = inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:length"
            ]

            belly_height = inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:height"
            ]

            added_wet_area = (
                belly_length * belly_width
                + 2.0 * belly_length * belly_height
                + 2.0 * belly_height * belly_width
            )

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
            ] = (belly_length + 2.0 * belly_height) / wet_area * cd0_fus

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:height",
            ] = (2.0 * belly_width + 2.0 * belly_length) / wet_area * cd0_fus

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":dimension:length",
            ] = (belly_width + 2.0 * belly_height) / wet_area * cd0_fus

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:geometry:fuselage:wet_area",
            ] = -added_wet_area / wet_area**2.0 * cd0_fus

            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
            ] = added_wet_area / wet_area

        else:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":dimension:width",
            ] = 0.0
