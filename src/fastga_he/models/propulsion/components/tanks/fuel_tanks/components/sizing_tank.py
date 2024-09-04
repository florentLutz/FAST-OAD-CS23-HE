# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_tank_unusable_fuel import SizingFuelTankUnusableFuel
from .sizing_tank_total_fuel_mission import SizingFuelTankTotalFuelMission
from .sizing_tank_volume import SizingFuelTankVolume
from .sizing_tank_cg_x import SizingFuelTankCGX
from .sizing_tank_cg_y import SizingFuelTankCGY
from .sizing_tank_length import SizingFuelTankLength
from .sizing_tank_height import SizingFuelTankHeight
from .sizing_tank_width import SizingFuelTankWidth
from .sizing_tank_weight import SizingFuelTankWeight
from .sizing_tank_drag import SizingFuelTankDrag
from .sizing_tank_prep_for_loads import SizingFuelTankPreparationForLoads

from .cstr_fuel_tank import ConstraintsFuelTank

from ..constants import POSSIBLE_POSITION


class SizingFuelTank(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the fuel tank.
    """

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the fuel tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="unusable_fuel",
            subsys=SizingFuelTankUnusableFuel(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="total_fuel",
            subsys=SizingFuelTankTotalFuelMission(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="constraints_tank",
            subsys=ConstraintsFuelTank(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_x",
            subsys=SizingFuelTankCGX(fuel_tank_id=fuel_tank_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_cg_y",
            subsys=SizingFuelTankCGY(fuel_tank_id=fuel_tank_id, position=position),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_volume",
            subsys=SizingFuelTankVolume(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_length",
            subsys=SizingFuelTankLength(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_height",
            subsys=SizingFuelTankHeight(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_width",
            subsys=SizingFuelTankWidth(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_weight",
            subsys=SizingFuelTankWeight(fuel_tank_id=fuel_tank_id),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "tank_drag_ls" if low_speed_aero else "tank_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingFuelTankDrag(
                    fuel_tank_id=fuel_tank_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )

        if position == "inside_the_wing":
            self.add_subsystem(
                name="preparation_for_loads",
                subsys=SizingFuelTankPreparationForLoads(
                    fuel_tank_id=fuel_tank_id,
                    position=position,
                ),
                promotes=["*"],
            )
