# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.sizing_weight import SizingGearboxWeight
from ..components.sizing_dimension_scaling import SizingGearboxDimensionScaling
from ..components.sizing_dimension import SizingGearboxDimensions
from ..components.sizing_cg_x import SizingGearboxCGX
from ..components.sizing_cg_y import SizingGearboxCGY
from ..components.sizing_drag import SizingGearboxDrag

from .cstr_gearbox import ConstraintsGearbox

from ..constants import POSSIBLE_POSITION


class SizingGearbox(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the gearbox.
    """

    def initialize(self):
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the gearbox, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        position = self.options["position"]
        gearbox_id = self.options["gearbox_id"]

        self.add_subsystem(
            "constraints",
            ConstraintsGearbox(gearbox_id=gearbox_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "dimension_scaling",
            SizingGearboxDimensionScaling(gearbox_id=gearbox_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "dimensions",
            SizingGearboxDimensions(gearbox_id=gearbox_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "gearbox_weight",
            SizingGearboxWeight(gearbox_id=gearbox_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="gearbox_CG_x",
            subsys=SizingGearboxCGX(gearbox_id=gearbox_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="gearbox_CG_y",
            subsys=SizingGearboxCGY(gearbox_id=gearbox_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "gearbox_drag_ls" if low_speed_aero else "gearbox_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingGearboxDrag(
                    gearbox_id=gearbox_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
