# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingFuelSystemCGY(om.ExplicitComponent):
    """
    Class that computes the Y-CG of the hydrogen fuel system based on its position. Will be based on simple
    geometric ratios, no consideration of volume will be implemented for now.
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="from_rear_to_center",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the hydrogen fuel system, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        position = self.options["position"]
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        # At least one input is needed regardless of the case
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y",
            units="m",
            val=0.0,
            desc="Y position of the ICE center of gravity",
        )
        wing_related = (
            position == "from_rear_to_wing"
            or position == "from_center_to_wing"
            or position == "from_center_to_front"
        )

        if position == "in_the_wing" or wing_related:
            self.add_input(
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":CG:y_ratio",
                val=np.nan,
                desc="Y position of the power source center of gravity as a ratio of the wing "
                "half-span",
            )

            self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        position = self.options["position"]
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        half_span = 0.5 * inputs["data:geometry:wing:span"]
        y_ratio = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y_ratio"
        ]
        wing_related = (
            position == "from_rear_to_wing"
            or position == "from_center_to_wing"
            or position == "from_center_to_front"
        )

        if position == "in_the_wing":
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y"
            ] = half_span * y_ratio
        elif wing_related:
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y"
            ] = 0.5 * half_span * y_ratio

        else:
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y"
            ] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y_ratio"
        ]
        wing_related = (
            position == "from_rear_to_wing"
            or position == "from_center_to_wing"
            or position == "from_center_to_front"
        )

        if position == "in_the_wing":
            partials[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y",
                "data:geometry:wing:span",
            ] = 0.5 * y_ratio
            partials[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y",
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":CG:y_ratio",
            ] = 0.5 * span

        elif wing_related:
            partials[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y",
                "data:geometry:wing:span",
            ] = 0.25 * y_ratio
            partials[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y",
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":CG:y_ratio",
            ] = 0.25 * span
