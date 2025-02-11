# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingGaseousHydrogenTankCGX(om.ExplicitComponent):
    """
    Class that computes the CG in X-direction of the tank according to the position given in the
    options.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the gaseous hydrogen tank, "
            "possible position include " + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":CG:x",
            units="m",
            val=2.8,
            desc="X position of the tank center of gravity",
        )

        if position == "wing_pod":
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

            self.declare_partials(of="*", wrt="data:geometry:wing:MAC:at25percent:x", val=1)
            self.declare_partials(of="*", wrt="data:geometry:wing:MAC:length", val=0.25)

        elif position == "in_the_back":
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
            self.add_input(
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length",
                units="m",
                val=np.nan,
                desc="Length of the tank, as in the size of the tank along the X-axis",
            )

            self.declare_partials(of="*", wrt="data:geometry:fuselage:front_length", val=1.0)
            self.declare_partials(of="*", wrt="data:geometry:cabin:length", val=1.0)
            self.declare_partials(
                of="*",
                wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length",
                val=0.5,
            )
        # We can do an else for the last option since we gave OpenMDAO the possible, ensuring it
        # is one among them
        else:
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

            self.declare_partials(of="*", wrt="data:geometry:fuselage:front_length", val=1.0)
            self.declare_partials(of="*", wrt="data:geometry:cabin:length", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]

        if position == "wing_pod":
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":CG:x"
            ] = (
                inputs["data:geometry:wing:MAC:at25percent:x"]
                + 0.25 * inputs["data:geometry:wing:MAC:length"]
            )

        elif position == "in_the_back":
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":CG:x"
            ] = (
                inputs["data:geometry:fuselage:front_length"]
                + inputs["data:geometry:cabin:length"]
                + 0.5
                * inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":dimension:length"
                ]
            )

        else:
            outputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":CG:x"
            ] = (
                inputs["data:geometry:fuselage:front_length"]
                + 0.5 * inputs["data:geometry:cabin:length"]
            )
