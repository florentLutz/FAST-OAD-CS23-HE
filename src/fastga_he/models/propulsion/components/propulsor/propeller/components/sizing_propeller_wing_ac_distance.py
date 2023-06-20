# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingPropellerWingACDistance(om.ExplicitComponent):
    """
    Computation of the distance between the propeller and the wing aerodynamic center
    """

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the propeller, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        # TODO: All those formulas (including the one in other components) should include the
        #  sweep at some point

        position = self.options["position"]
        propeller_id = self.options["propeller_id"]

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_AC",
            val=np.nan,
            units="m",
            desc="Distance between the propeller and the wing aerodynamic center",
        )

        if position == "on_the_wing":

            self.add_input(
                name="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE",
                val=np.nan,
                units="m",
                desc="Distance between the propeller and the leading edge",
            )
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_AC",
                wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE",
                val=1.0,
            )
            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_AC",
                wrt="data:geometry:wing:MAC:length",
                val=0.25,
            )

        else:

            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_AC",
                wrt="data:geometry:wing:MAC:at25percent:x",
                val=1.0,
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        position = self.options["position"]
        propeller_id = self.options["propeller_id"]

        if position == "on_the_wing":

            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_AC"
            ] = (
                inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE"]
                + 0.25 * inputs["data:geometry:wing:MAC:length"]
            )

        else:

            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_AC"
            ] = inputs["data:geometry:wing:MAC:at25percent:x"]
