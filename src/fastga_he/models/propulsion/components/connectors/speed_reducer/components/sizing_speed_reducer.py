# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..components.sizing_weight import SizingSpeedReducerWeight
from ..components.sizing_dimension_scaling import SizingSpeedReducerDimensionScaling
from ..components.sizing_dimension import SizingSpeedReducerDimensions
from ..components.sizing_cg_x import SizingSpeedReducerCGX
from ..components.sizing_cg_y import SizingSpeedReducerCGY
from ..components.sizing_drag import SizingSpeedReducerDrag

from .cstr_speed_reducer import ConstraintsSpeedReducer

from ..constants import POSSIBLE_POSITION


class SizingSpeedReducer(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the DC/DC converter.
    """

    def initialize(self):
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the speed reducer, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        position = self.options["position"]
        speed_reducer_id = self.options["speed_reducer_id"]

        self.add_subsystem(
            "constraints",
            ConstraintsSpeedReducer(speed_reducer_id=speed_reducer_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "dimension_scaling",
            SizingSpeedReducerDimensionScaling(speed_reducer_id=speed_reducer_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "dimensions",
            SizingSpeedReducerDimensions(speed_reducer_id=speed_reducer_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "reducer_weight",
            SizingSpeedReducerWeight(speed_reducer_id=speed_reducer_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="reducer_CG_x",
            subsys=SizingSpeedReducerCGX(speed_reducer_id=speed_reducer_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="reducer_CG_y",
            subsys=SizingSpeedReducerCGY(speed_reducer_id=speed_reducer_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "reducer_drag_ls" if low_speed_aero else "reducer_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingSpeedReducerDrag(
                    speed_reducer_id=speed_reducer_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
