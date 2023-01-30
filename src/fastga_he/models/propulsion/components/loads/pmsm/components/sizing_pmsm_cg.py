# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingPMSMCG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the pmsm, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        motor_id = self.options["motor_id"]
        position = self.options["position"]

        if position == "on_the_wing":

            self.add_input(
                name="data:propulsion:he_power_train:PMSM:" + motor_id + ":from_LE",
                val=np.nan,
                units="m",
                desc="Distance between the PMSM front face and the leading edge",
            )
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

            self.add_input(
                name="data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
                val=np.nan,
                units="m",
            )

        else:

            self.add_input(
                name="data:propulsion:he_power_train:PMSM:" + motor_id + ":front_length_ratio",
                val=np.nan,
                desc="Location of the PMSM CG as a ratio of the aircraft front length",
            )
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the PMSM center of gravity",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]
        position = self.options["position"]

        if position == "on_the_wing":

            distance_from_le = inputs[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":from_LE"
            ]
            motor_length = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":length"]
            l0_wing = inputs["data:geometry:wing:MAC:length"]
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

            outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x"] = (
                fa_length - 0.25 * l0_wing - distance_from_le + 0.5 * motor_length
            )

        else:

            lav = inputs["data:geometry:fuselage:front_length"]
            lav_ratio = inputs[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":front_length_ratio"
            ]

            outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x"] = lav * lav_ratio

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]
        position = self.options["position"]

        if position == "on_the_wing":

            partials[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":from_LE",
            ] = -1
            partials[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x",
                "data:geometry:wing:MAC:length",
            ] = -0.25
            partials[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
            ] = 0.5

        else:

            partials[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":front_length_ratio",
            ] = inputs["data:geometry:fuselage:front_length"]
            partials[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":front_length_ratio"]
