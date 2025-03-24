# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingH2FuelSystemLength(om.ExplicitComponent):
    """
    Computation of the hydrogen fuel system pipe network length based on the position of its source
    and target.
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
            default="in_the_middle",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the hydrogen fuel system, possible position "
            "include " + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare(
            name="wing_related",
            default=False,
            types=bool,
            desc="Option identifies weather the system reaches inside the wing or not",
        )
        self.options.declare(
            name="compact",
            default=False,
            types=bool,
            desc="Option identifies weather the system is installed compactly in one position",
        )

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        wing_related = self.options["wing_related"]
        position = self.options["position"]
        compact = self.options["compact"]

        if compact:
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
            self.declare_partials(of="*", wrt="*", val=1.0)
        elif position == "in_the_middle":
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
            self.declare_partials(of="*", wrt="*", val=1.0)
        else:
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

            if wing_related:
                self.add_input("data:geometry:wing:span", val=np.nan, units="m")
                self.add_input(
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":CG:y_ratio",
                    val=0.5,
                    desc="Y position of the power source center of gravity as a ratio of the wing "
                    "half-span",
                )
                self.declare_partials("*", "*", method="exact")

            self.declare_partials("*", "data:geometry:cabin:length", val=0.5)

        self.add_output(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":dimension:length",
            units="m",
            val=5.0,
            desc="Total length of the h2 fuel system",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        wing_related = self.options["wing_related"]
        position = self.options["position"]
        compact = self.options["compact"]

        if compact:
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":dimension:length"
            ] = inputs["data:geometry:wing:MAC:length"]

        elif position == "in_the_middle":
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":dimension:length"
            ] = inputs["data:geometry:cabin:length"]
        else:
            l_in_fus = 0.5 * inputs["data:geometry:cabin:length"]
            if wing_related:
                outputs[
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":dimension:length"
                ] = (
                    l_in_fus
                    + 0.5
                    * inputs["data:geometry:wing:span"]
                    * inputs[
                        "data:propulsion:he_power_train:H2_fuel_system:"
                        + h2_fuel_system_id
                        + ":CG:y_ratio"
                    ]
                )
            else:
                outputs[
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":dimension:length"
                ] = l_in_fus

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        wing_related = self.options["wing_related"]
        position = self.options["position"]
        compact = self.options["compact"]

        if not compact and not position == "in_thw_middle" and wing_related:
            partials[
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":dimension:length",
                "data:geometry:wing:span",
            ] = (
                0.5
                * inputs[
                    "data:propulsion:he_power_train:H2_fuel_system:"
                    + h2_fuel_system_id
                    + ":CG:y_ratio"
                ]
            )

            partials[
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":dimension:length",
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":CG:y_ratio",
            ] = 0.5 * inputs["data:geometry:wing:span"]
