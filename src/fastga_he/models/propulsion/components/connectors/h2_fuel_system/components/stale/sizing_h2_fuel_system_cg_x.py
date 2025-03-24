# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingH2FuelSystemCGX(om.ExplicitComponent):
    """
    Computation of the hydrogen fuel system X-CG based on the positions of the storage and the
    source components connected to the hydrogen fuel system.
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
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        wing_related = (
            position == "from_front_to_wing"
            or position == "from_rear_to_wing"
            or position == "from_center_to_wing"
        )

        self.add_output(
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the hydrogen fuel system center of gravity",
        )
        if position == "in_the_wing" or wing_related:
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        if wing_related:
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")
            self.add_input(
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":CG:y_ratio",
                val=np.nan,
                desc="Y position of the power source center of gravity as a ratio of the wing "
                "half-span",
            )
            self.declare_partials("*", "*", method="exact")

        if position == "from_center_to_front":
            self.declare_partials("*", "data:geometry:cabin:length", val=0.25)

        if position == "at_center" or position == "from_rear_to_front":
            self.declare_partials("*", "data:geometry:cabin:length", val=0.5)

        if position == "from_rear_to_center":
            self.declare_partials("*", "data:geometry:cabin:length", val=0.75)

        if position == "in_the_rear":
            self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
            self.declare_partials("*", "data:geometry:cabin:length", val=1.0)
            self.declare_partials("*", "data:geometry:fuselage:rear_length", val=0.5)

        if not (position == "in_the_wing" or wing_related):
            self.declare_partials(of="*", wrt="data:geometry:fuselage:front_length", val=1.0)

        if position == "in_the_wing" or position == "from_center_to_wing":
            self.declare_partials("*", "data:geometry:wing:MAC:at25percent:x", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]
        front_length = inputs["data:geometry:fuselage:front_length"]
        cabin_length = inputs["data:geometry:cabin:length"]

        wing_related = (
            position == "from_front_to_wing"
            or position == "from_rear_to_wing"
            or position == "from_center_to_wing"
        )

        if position == "at_center" or position == "from_rear_to_front":
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
            ] = front_length + 0.5 * cabin_length

        elif position == "in_the_rear":
            rear_length = inputs["data:geometry:fuselage:rear_length"]
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
            ] = front_length + cabin_length + 0.5 * rear_length

        elif position == "from_rear_to_center":
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
            ] = front_length + 0.75 * cabin_length

        elif position == "from_center_to_front":
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
            ] = front_length + 0.25 * cabin_length

        if position == "in_the_wing" or wing_related:
            mac25x = inputs["data:geometry:wing:MAC:at25percent:x"]
            if position == "in_the_wing" or position == "from_center_to_wing":
                outputs[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
                ] = mac25x

            elif wing_related:
                span = inputs["data:geometry:wing:span"]
                y_ratio = inputs[
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":CG:y_ratio"
                ]
                total_length = 0.5 * (cabin_length + span * y_ratio)

                if position == "from_front_to_wing":
                    cg_distance = mac25x - 0.25 * cabin_length - front_length
                    outputs[
                        "data:propulsion:he_power_train:H2_fuel_system:"
                        + h2_fuel_system_id
                        + ":CG:x"
                    ] = mac25x - cg_distance * 0.5 * cabin_length / total_length
                if position == "from_rear_to_wing":
                    cg_distance = 0.75 * cabin_length + front_length - mac25x
                    outputs[
                        "data:propulsion:he_power_train:H2_fuel_system:"
                        + h2_fuel_system_id
                        + ":CG:x"
                    ] = mac25x + cg_distance * 0.5 * cabin_length / total_length

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]
        front_length = inputs["data:geometry:fuselage:front_length"]
        cabin_length = inputs["data:geometry:cabin:length"]

        if position == "from_front_to_wing" or position == "from_rear_to_wing":
            span = inputs["data:geometry:wing:span"]
            mac25x = inputs["data:geometry:wing:MAC:at25percent:x"]
            y_ratio = inputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:y_ratio"
            ]
            total_length = 0.5 * (cabin_length + span * y_ratio)

            partials[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0 - 0.5 * cabin_length / total_length

            partials[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 0.5 * cabin_length / total_length

            if position == "from_front_to_wing":
                cg_distance = mac25x - 0.25 * cabin_length - front_length

                partials[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                    "data:geometry:wing:span",
                ] = cg_distance * cabin_length * y_ratio / (2.0 * total_length) ** 2.0

                partials[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":CG:y_ratio",
                ] = cg_distance * cabin_length * span / (2.0 * total_length) ** 2.0

                partials[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                    "data:geometry:cabin:length",
                ] = (
                    0.25 * cabin_length**2.0
                    + (span * y_ratio) * (0.5 * cabin_length + front_length - mac25x)
                ) / (2.0 * total_length) ** 2.0

            if position == "from_rear_to_wing":
                cg_distance = 0.75 * cabin_length + front_length - mac25x

                partials[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                    "data:geometry:wing:span",
                ] = -cg_distance * cabin_length * y_ratio / (2.0 * total_length) ** 2.0

                partials[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":CG:y_ratio",
                ] = -cg_distance * cabin_length * span / (2.0 * total_length) ** 2.0

                partials[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
                    "data:geometry:cabin:length",
                ] = (
                    0.75 * cabin_length**2.0
                    + (span * y_ratio) * (1.5 * cabin_length + front_length - mac25x)
                ) / (2.0 * total_length) ** 2.0
