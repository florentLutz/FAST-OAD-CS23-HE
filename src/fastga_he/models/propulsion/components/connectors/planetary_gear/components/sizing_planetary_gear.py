# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.sizing_weight import SizingPlanetaryGearWeight
from ..components.sizing_dimension_scaling import SizingPlanetaryGearDimensionScaling
from ..components.sizing_dimension import SizingPlanetaryGearDimensions
from ..components.sizing_cg_x import SizingPlanetaryGearCGX
from ..components.sizing_cg_y import SizingPlanetaryGearCGY
from ..components.sizing_drag import SizingPlanetaryGearDrag

from .cstr_planetary_gear import ConstraintsPlanetaryGear

from ..constants import POSSIBLE_POSITION


class SizingPlanetaryGear(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the planetary gearbox.
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

        # The followong option(s) is/are dummy option(s) to ensure compatibility
        self.options.declare(
            "gear_mode",
            default="percent_split",
            desc="Mode of the planetary gear, should be either percent_split or power_share",
            values=["percent_split", "power_share"],
        )

    def setup(self):
        position = self.options["position"]
        planetary_gear_id = self.options["planetary_gear_id"]

        self.add_subsystem(
            "constraints",
            ConstraintsPlanetaryGear(planetary_gear_id=planetary_gear_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "dimension_scaling",
            SizingPlanetaryGearDimensionScaling(planetary_gear_id=planetary_gear_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "dimensions",
            SizingPlanetaryGearDimensions(planetary_gear_id=planetary_gear_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "gearbox_weight",
            SizingPlanetaryGearWeight(planetary_gear_id=planetary_gear_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="gearbox_CG_x",
            subsys=SizingPlanetaryGearCGX(planetary_gear_id=planetary_gear_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="gearbox_CG_y",
            subsys=SizingPlanetaryGearCGY(planetary_gear_id=planetary_gear_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "gearbox_drag_ls" if low_speed_aero else "gearbox_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingPlanetaryGearDrag(
                    planetary_gear_id=planetary_gear_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
