# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .cstr_turbo_generator import ConstraintsTurboGenerator

from .sizing_weight import SizingTurboGeneratorWeight
from .sizing_turbo_generator_cg_x import SizingTurboGeneratorCGX
from .sizing_turbo_generator_cg_y import SizingTurboGeneratorCGY
from .sizing_turbo_generator_drag import SizingTurboGeneratorDrag

from ..constants import POSSIBLE_POSITION


class SizingTurboGenerator(om.Group):
    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the turbo generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        turbo_generator_id = self.options["turbo_generator_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="constraints_turbo_generator",
            subsys=ConstraintsTurboGenerator(turbo_generator_id=turbo_generator_id),
            promotes=["*"],
        )

        self.add_subsystem(
            "weight",
            SizingTurboGeneratorWeight(turbo_generator_id=turbo_generator_id),
            promotes=["data:*"],
        )

        self.add_subsystem(
            "generator_cg_x",
            SizingTurboGeneratorCGX(turbo_generator_id=turbo_generator_id, position=position),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "generator_cg_y",
            SizingTurboGeneratorCGY(turbo_generator_id=turbo_generator_id, position=position),
            promotes=["data:*"],
        )

        for low_speed_aero in [True, False]:
            system_name = (
                "turbo_generator_drag_ls" if low_speed_aero else "turbo_generator_drag_cruise"
            )
            self.add_subsystem(
                name=system_name,
                subsys=SizingTurboGeneratorDrag(
                    turbo_generator_id=turbo_generator_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
