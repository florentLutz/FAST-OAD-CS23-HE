# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .sizing_tank_unusable_hydrogen import SizingGaseousHydrogenTankUnusableHydrogen
from .sizing_tank_total_hydrogen_mission import SizingGaseousHydrogenTankTotalHydrogenMission
from .sizing_tank_wall_thickness import SizingGaseousHydrogenTankWallThickness
from .sizing_tank_cg_x import SizingGaseousHydrogenTankCGX
from .sizing_tank_cg_y import SizingGaseousHydrogenTankCGY
from .sizing_tank_length import SizingGaseousHydrogenTankLength
from .sizing_tank_inner_volume import SizingGaseousHydrogenTankInnerVolume
from .sizing_tank_inner_diameter import SizingGaseousHydrogenTankInnerDiameter
from .sizing_tank_weight import SizingGaseousHydrogenTankWeight
from .sizing_tank_gravimetric_index import SizingGaseousHydrogenTankGravimetricIndex
from .sizing_tank_drag import SizingGaseousHydrogenTankDrag
from .sizing_tank_outer_diameter import SizingGaseousHydrogenTankOuterDiameter
from .sizing_tank_length_fuselage_contraints import (
    SizingGaseousHydrogenTankLengthFuselageConstraints,
)

from .cstr_gaseous_hydrogen_tank import ConstraintsGaseousHydrogenTank

from ..constants import POSSIBLE_POSITION


class SizingGaseousHydrogenTank(om.Group):
    """
    Class that regroups all the subcomponents for the sizing of the gaseous hydrogen tank.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_cabin",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the gaseous hydrogen tank, "
            "possible position include " + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="unusable_gaseous_hydrogen",
            subsys=SizingGaseousHydrogenTankUnusableHydrogen(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="total_gaseous_hydrogen",
            subsys=SizingGaseousHydrogenTankTotalHydrogenMission(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_constraints",
            subsys=ConstraintsGaseousHydrogenTank(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_volume",
            subsys=SizingGaseousHydrogenTankInnerVolume(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_outer_diameter",
            subsys=SizingGaseousHydrogenTankOuterDiameter(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_diameter",
            subsys=SizingGaseousHydrogenTankInnerDiameter(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_wall_thickness",
            subsys=SizingGaseousHydrogenTankWallThickness(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_length",
            subsys=SizingGaseousHydrogenTankLength(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_x",
            subsys=SizingGaseousHydrogenTankCGX(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_y",
            subsys=SizingGaseousHydrogenTankCGY(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_length_fuselage_constraints",
            subsys=SizingGaseousHydrogenTankLengthFuselageConstraints(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id,
                position=position,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_weight",
            subsys=SizingGaseousHydrogenTankWeight(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_gravimetric_index",
            subsys=SizingGaseousHydrogenTankGravimetricIndex(
                gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "tank_drag_ls" if low_speed_aero else "tank_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingGaseousHydrogenTankDrag(
                    gaseous_hydrogen_tank_id=gaseous_hydrogen_tank_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
