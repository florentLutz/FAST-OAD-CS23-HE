# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingFuelSystemCGX(om.ExplicitComponent):
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

        if not position == "from_front_to_wing":
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the hydrogen fuel system center of gravity",
        )

        if position == "in_the_wing":
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        if (
            position == "from_front_to_wing"
            or position == "from_rear_to_wing"
            or position == "from_center_to_wing"
        ):
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")
            self.add_input(
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":CG:y_ratio",
                val=np.nan,
                desc="Y position of the power source center of gravity as a ratio of the wing "
                "half-span",
            )

        if position == "in_the_back":
            self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        position = self.options["position"]

        if position == "in_the_wing":
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

            outputs[
                "data:propulsion:he_power_train:H2_fuel_system:" + h2_fuel_system_id + ":CG:x"
            ] = fa_length

        elif position == "in_the_back":
            front_length = inputs["data:geometry:fuselage:front_length"]

            outputs["data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":CG:x"] = (
                front_length
            )

        else:
            front_length = inputs["data:geometry:fuselage:front_length"]
            cabin_length = inputs["data:geometry:cabin:length"]

            outputs["data:propulsion:he_power_train:fuel_system:" + h2_fuel_system_id + ":CG:x"] = (
                front_length + cabin_length
            )
