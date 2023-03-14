# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingGeneratorDrag(om.ExplicitComponent):
    """
    Class that computes the drag coefficient of the generator based on its position. Will be 0.0
    all the time as we wil make the assumption that the generator is "inside" the fuselage or
    when in the nose, it will be computed as not contributing, just like we did it for ICE,
    turboprop.
    """

    def initialize(self):
        self.options.declare(
            name="generator_id",
            default=None,
            desc="Identifier of the generator",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        # For refractoring purpose we just match the option to the tag in the variable name and
        # use it
        generator_id = self.options["generator_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":" + ls_tag + ":CD0",
            val=0.0,
        )
