# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingGeneratorCGX(om.ExplicitComponent):
    """
    Computation of the position, along the x-axis of the CG of the generator. If the generator is
    inside the wing we will consider it at the quarter chord
    """

    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        generator_id = self.options["generator_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the generator center of gravity",
        )

        if position == "inside_the_wing":

            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        elif position == "in_the_front":

            self.add_input(
                name="data:propulsion:he_power_train:generator:" + generator_id + ":length",
                val=np.nan,
                units="m",
            )
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        else:

            self.add_input(
                name="data:propulsion:he_power_train:generator:" + generator_id + ":length",
                val=np.nan,
                units="m",
            )
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        generator_id = self.options["generator_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            outputs["data:propulsion:he_power_train:generator:" + generator_id + ":CG:x"] = inputs[
                "data:geometry:wing:MAC:at25percent:x"
            ]

        elif position == "in_the_front":

            lav = inputs["data:geometry:fuselage:front_length"]
            generator_length = inputs[
                "data:propulsion:he_power_train:generator:" + generator_id + ":length"
            ]

            outputs["data:propulsion:he_power_train:generator:" + generator_id + ":CG:x"] = (
                lav - generator_length / 2.0
            )

        else:

            cabin_length = inputs["data:geometry:cabin:length"]
            lav = inputs["data:geometry:fuselage:front_length"]
            generator_length = inputs[
                "data:propulsion:he_power_train:generator:" + generator_id + ":length"
            ]

            outputs["data:propulsion:he_power_train:generator:" + generator_id + ":CG:x"] = (
                lav + cabin_length + generator_length / 2.0
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        generator_id = self.options["generator_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            partials[
                "data:propulsion:he_power_train:generator:" + generator_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0

        elif position == "in_the_front":

            partials[
                "data:propulsion:he_power_train:generator:" + generator_id + ":CG:x",
                "data:propulsion:he_power_train:generator:" + generator_id + ":length",
            ] = (
                -1.0 / 2.0
            )
            partials[
                "data:propulsion:he_power_train:generator:" + generator_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 1.0

        else:
            partials[
                "data:propulsion:he_power_train:generator:" + generator_id + ":CG:x",
                "data:propulsion:he_power_train:generator:" + generator_id + ":length",
            ] = (
                1.0 / 2.0
            )
            partials[
                "data:propulsion:he_power_train:generator:" + generator_id + ":CG:x",
                "data:geometry:fuselage:front_length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:generator:" + generator_id + ":CG:x",
                "data:geometry:cabin:length",
            ] = 1.0
