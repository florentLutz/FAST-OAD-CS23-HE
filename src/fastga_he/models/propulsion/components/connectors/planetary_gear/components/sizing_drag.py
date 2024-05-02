# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingPlanetaryGearDrag(om.ExplicitComponent):
    """
    Class that computes the drag coefficient of a planetary gearbox based on its position. Will
    be 0.0 all the time as we will make the assumption that the gearbox is "inside" the
    fuselage or in the fairing of the motor if inside the wing (and thus is already accounted for
    in the wing drag calculation).
    """

    def initialize(self):
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the planetary gear, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        # For refractoring purpose we just match the option to the tag in the variable name and
        # use it
        planetary_gear_id = self.options["planetary_gear_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_output(
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":"
            + ls_tag
            + ":CD0",
            val=0.0,
        )
