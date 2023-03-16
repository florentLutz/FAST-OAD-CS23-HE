# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingICECG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        ice_id = self.options["ice_id"]
        position = self.options["position"]

        self.add_input(
            name="data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length",
            val=np.nan,
            units="m",
        )

        if position == "on_the_wing":

            self.add_input(
                name="data:propulsion:he_power_train:ICE:" + ice_id + ":from_LE",
                val=np.nan,
                units="m",
                desc="Distance between the ICE front face and the leading edge",
            )
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        elif position == "in_the_back":

            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the ICE center of gravity",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]
        position = self.options["position"]

        motor_length = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length"]

        if position == "on_the_wing":

            distance_from_le = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":from_LE"]
            l0_wing = inputs["data:geometry:wing:MAC:length"]
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

            outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x"] = (
                fa_length - 0.25 * l0_wing - distance_from_le + 0.5 * motor_length
            )

        elif position == "in_the_front":

            outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x"] = 0.5 * motor_length

        else:

            front_length = inputs["data:geometry:fuselage:front_length"]
            cabin_length = inputs["data:geometry:cabin:length"]

            outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x"] = (
                front_length + cabin_length + 0.5 * motor_length
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ice_id = self.options["ice_id"]
        position = self.options["position"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":nacelle:length",
        ] = 0.5

        if position == "on_the_wing":

            partials[
                "data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x",
                "data:propulsion:he_power_train:ICE:" + ice_id + ":from_LE",
            ] = -1
            partials[
                "data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x",
                "data:geometry:wing:MAC:length",
            ] = -0.25
            partials[
                "data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0

        elif position == "in_the_back":

            partials[
                "data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:ICE:" + ice_id + ":CG:x",
                "data:geometry:cabin:length",
            ] = 1.0
