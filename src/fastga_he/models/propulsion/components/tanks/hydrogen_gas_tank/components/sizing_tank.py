# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_tank_unusable_hydrogen import SizingHydrogenGasTankUnusableHydrogen
from .sizing_tank_total_hydrogen_mission import SizingHydrogenGasTankTotalHydrogenMission
from .sizing_tank_wall_thickness import SizingHydrogenGasTankWallThickness
from .sizing_tank_cg_x import SizingHydrogenGasTankCGX
from .sizing_tank_cg_y import SizingHydrogenGasTankCGY
from .sizing_tank_length import SizingHydrogenGasTankLength
from .sizing_tank_inner_volume import SizingHydrogenGasTankInnerVolume
from .sizing_tank_inner_diameter import SizingHydrogenGasTankInnerDiameter
from .sizing_tank_weight import SizingHydrogenGasTankWeight
from .sizing_gravimetric_index import SizingHydrogenGasTankGravimetricIndex
from .sizing_tank_drag import SizingHydrogenGasTankDrag
from .sizing_tank_outer_diameter import SizingHydrogenGasTankOuterDiameter
from .sizing_tank_overall_length import SizingHydrogenGasTankOverallLength
from .sizing_tank_overall_length_fuselage_check import (
    SizingHydrogenGasTankOverallLengthFuselageCheck,
)

from .cstr_hydrogen_gas_tank import ConstraintsHydrogenGasTank

from ..constants import POSSIBLE_POSITION


class SizingHydrogenGasTank(om.Group):
    """
    Class that regroups all the subcomponents for the sizing of the hydrogen gas tank.
    """

    def initialize(self):
        self.options.declare(
            name="hydrogen_gas_tank_id",
            default=None,
            desc="Identifier of the hydrogen gas tank",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_fuselage",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the hydrogen gas tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="unusable_hydrogen_gas",
            subsys=SizingHydrogenGasTankUnusableHydrogen(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="total_hydrogen_gas",
            subsys=SizingHydrogenGasTankTotalHydrogenMission(
                hydrogen_gas_tank_id=hydrogen_gas_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_constraints",
            subsys=ConstraintsHydrogenGasTank(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_volume",
            subsys=SizingHydrogenGasTankInnerVolume(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_outer_diameter",
            subsys=SizingHydrogenGasTankOuterDiameter(
                hydrogen_gas_tank_id=hydrogen_gas_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_diameter",
            subsys=SizingHydrogenGasTankInnerDiameter(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_wall_thickness",
            subsys=SizingHydrogenGasTankWallThickness(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_length",
            subsys=SizingHydrogenGasTankLength(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_overall_length",
            subsys=SizingHydrogenGasTankOverallLength(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_x",
            subsys=SizingHydrogenGasTankCGX(
                hydrogen_gas_tank_id=hydrogen_gas_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_y",
            subsys=SizingHydrogenGasTankCGY(
                hydrogen_gas_tank_id=hydrogen_gas_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_overall_length_length_fuselage_check",
            subsys=SizingHydrogenGasTankOverallLengthFuselageCheck(
                hydrogen_gas_tank_id=hydrogen_gas_tank_id,
                position=position,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_weight",
            subsys=SizingHydrogenGasTankWeight(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_gravimetric_index",
            subsys=SizingHydrogenGasTankGravimetricIndex(hydrogen_gas_tank_id=hydrogen_gas_tank_id),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "tank_drag_ls" if low_speed_aero else "tank_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingHydrogenGasTankDrag(
                    hydrogen_gas_tank_id=hydrogen_gas_tank_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
