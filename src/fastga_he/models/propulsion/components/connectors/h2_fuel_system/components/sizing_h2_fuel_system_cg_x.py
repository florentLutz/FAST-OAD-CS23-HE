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

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]
        wing_related = self.options["wing_related"]
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the hydrogen fuel system center of gravity",
        )

        self.declare_partials("*", "data:geometry:fuselage:front_length", val=1.0)
        if position == "in_the_middle":
            self.declare_partials("*", "data:geometry:cabin:length", val=0.5)

        elif wing_related:
            if position == "in_the_front":
                self.declare_partials("*", "data:geometry:cabin:length", val=0.375)
            if position == "in_the_rear":
                self.declare_partials("*", "data:geometry:cabin:length", val=0.875)
        else:
            if position == "in_the_front":
                self.declare_partials("*", "data:geometry:cabin:length", val=0.25)
            if position == "in_the_rear":
                self.declare_partials("*", "data:geometry:cabin:length", val=0.75)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]
        wing_related = self.options["wing_related"]
        front_length = inputs["data:geometry:fuselage:front_length"]
        cabin_length = inputs["data:geometry:cabin:length"]

        if position == "in_the_middle":
            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
            ] = front_length + 0.5 * cabin_length

        elif wing_related:
            if position == "in_the_front":
                outputs[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
                ] = front_length + 0.375 * cabin_length
            if position == "in_the_rear":
                outputs[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
                ] = front_length + 0.875 * cabin_length
        else:
            if position == "in_the_front":
                outputs[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
                ] = front_length + 0.25 * cabin_length
            if position == "in_the_rear":
                outputs[
                    "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
                ] = front_length + 0.75 * cabin_length
