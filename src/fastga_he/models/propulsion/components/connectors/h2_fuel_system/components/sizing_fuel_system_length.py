# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingH2FuelSystemLength(om.ExplicitComponent):
    """
    The length of the hydrogen fuel system. The pipe inside the length
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:near",
            val=0.0,
            desc="Number of near connections for the h2 fuel system",
        )
        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:front",
            val=0.0,
            desc="Number of front connections for the h2 fuel system",
        )

        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:rear",
            val=0.0,
            desc="Number of rear connections for the h2 fuel system",
        )

        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:wing",
            val=0.0,
            desc="Number of wing connections for the h2 fuel system",
        )

        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y_ratio",
            val=0.5,
            desc="Y position of the power source center of gravity as a ratio of the wing "
            "half-span",
        )

        self.add_output(
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            units="m",
            val=5.0,
            desc="Total length of the h2 fuel system",
        )
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        num_front = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:front"
        ]
        num_rear = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:rear"
        ]
        num_wing = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:wing"
        ]
        num_near = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:near"
        ]
        half_span = 0.5 * inputs["data:geometry:wing:span"]
        half_l_cabin = 0.5 * inputs["data:geometry:cabin:length"]
        y_ratio = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y_ratio"
        ]
        wing_mac = inputs["data:geometry:wing:MAC:length"]

        outputs[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length"
        ] = (
            (num_front + num_rear) * half_l_cabin
            + num_wing * y_ratio * half_span
            + wing_mac * num_near
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        num_front = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:front"
        ]
        num_rear = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:rear"
        ]
        num_wing = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:wing"
        ]
        num_near = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:near"
        ]
        half_span = 0.5 * inputs["data:geometry:wing:span"]
        half_l_cabin = 0.5 * inputs["data:geometry:cabin:length"]
        y_ratio = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y_ratio"
        ]
        wing_mac = inputs["data:geometry:wing:MAC:length"]

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:front",
        ] = half_l_cabin

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:rear",
        ] = half_l_cabin

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:geometry:cabin:length",
        ] = 0.5 * (num_front + num_rear)

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:wing",
        ] = y_ratio * half_span

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y_ratio",
        ] = num_wing * half_span

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:geometry:wing:span",
        ] = 0.5 * num_wing * y_ratio

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":connection:near",
        ] = wing_mac

        partials[
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":length",
            "data:geometry:wing:MAC:length",
        ] = num_near
