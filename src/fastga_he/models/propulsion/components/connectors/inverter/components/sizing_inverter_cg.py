# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingInverterCG(om.ExplicitComponent):
    """
    Class that computes the CG of the inverter based on its position. Will be based on simple
    geometric ratios, no consideration of volume will be implemented for now.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the inverter, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        position = self.options["position"]
        inverter_id = self.options["inverter_id"]

        if position == "inside_the_wing":

            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        elif position == "in_the_front":

            self.add_input(
                name="data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":front_length_ratio",
                val=0.9,
                desc="Location of the inverter CG as a ratio of the aircraft front length",
            )
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        else:
            # Guaranteed to be in the back since we can only use defined position
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
            self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
            self.add_input(
                name="data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":rear_length_ratio",
                val=0.1,
                desc="Location of the inverter CG as a ratio of the aircraft rear length",
            )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the inverter center of gravity",
        )
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        position = self.options["position"]
        inverter_id = self.options["inverter_id"]

        if position == "inside_the_wing":

            outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x"] = inputs[
                "data:geometry:wing:MAC:at25percent:x"
            ]

        elif position == "in_the_front":

            outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x"] = (
                inputs[
                    "data:propulsion:he_power_train:inverter:" + inverter_id + ":front_length_ratio"
                ]
                * inputs["data:geometry:fuselage:front_length"]
            )

        else:

            outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x"] = (
                inputs["data:geometry:fuselage:front_length"]
                + inputs["data:geometry:cabin:length"]
                + inputs["data:geometry:fuselage:rear_length"]
                * inputs[
                    "data:propulsion:he_power_train:inverter:" + inverter_id + ":rear_length_ratio"
                ]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        position = self.options["position"]
        inverter_id = self.options["inverter_id"]

        if position == "inside_the_wing":

            partials[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0

        elif position == "in_the_front":

            partials[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":front_length_ratio",
            ] = inputs["data:geometry:fuselage:front_length"]
            partials[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":front_length_ratio"
            ]

        else:

            partials[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
                "data:geometry:cabin:length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
                "data:geometry:fuselage:rear_length",
            ] = inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":rear_length_ratio"
            ]
            partials[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":CG:x",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":rear_length_ratio",
            ] = inputs["data:geometry:fuselage:rear_length"]
