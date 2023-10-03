# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingFuelTankCGX(om.ExplicitComponent):
    """Class that computes the CG of the battery according to the position given in the options."""

    def initialize(self):

        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the fuel tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the battery center of gravity",
        )

        if position == "inside_the_wing" or position == "wing_pod":

            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

            self.declare_partials(of="*", wrt="data:geometry:wing:MAC:at25percent:x", val=1)
            self.declare_partials(of="*", wrt="data:geometry:wing:MAC:length", val=0.25)

        # We can do an else for the last option since we gave OpenMDAO the possible, ensuring it
        # is one among them
        else:

            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

            self.declare_partials(of="*", wrt="data:geometry:fuselage:front_length", val=1.0)
            self.declare_partials(of="*", wrt="data:geometry:cabin:length", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing" or position == "wing_pod":

            outputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:x"] = (
                inputs["data:geometry:wing:MAC:at25percent:x"]
                + 0.25 * inputs["data:geometry:wing:MAC:length"]
            )

        else:

            outputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:x"] = (
                inputs["data:geometry:fuselage:front_length"]
                + 0.5 * inputs["data:geometry:cabin:length"]
            )
