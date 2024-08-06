# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingPropellerDrag(om.ExplicitComponent):
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
        # Not as useful as the ones in aerodynamics, here it will just be run twice in the sizing
        # group
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        # For refractoring purpose we just match the option to the tag in the variable name and
        # use it
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        propeller_id = self.options["propeller_id"]

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":" + ls_tag + ":CD0",
            val=0.0,
        )
