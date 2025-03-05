# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.sizing_fuel_system_cg_x import SizingFuelSystemCGX
from ..components.sizing_fuel_system_cg_y import SizingFuelSystemCGY
from ..components.sizing_fuel_system_volume import SizingFuelSystemCapacityVolume
from ..components.sizing_fuel_system_weight import SizingFuelSystemWeight
from ..components.sizing_fuel_system_drag import SizingFuelSystemDrag

from ..constants import POSSIBLE_POSITION


class SizingFuelSystem(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the fuel system.
    """

    def initialize(self):
        self.options.declare(
            name="fuel_system_id",
            default=None,
            desc="Identifier of the fuel system",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the fuel system, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        fuel_system_id = self.options["fuel_system_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="fuel_system_cg_x",
            subsys=SizingFuelSystemCGX(fuel_system_id=fuel_system_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fuel_system_cg_y",
            subsys=SizingFuelSystemCGY(fuel_system_id=fuel_system_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="connected_tank_volume",
            subsys=SizingFuelSystemCapacityVolume(fuel_system_id=fuel_system_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fuel_system_mass",
            subsys=SizingFuelSystemWeight(fuel_system_id=fuel_system_id),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "fuel_system_drag_ls" if low_speed_aero else "fuel_system_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingFuelSystemDrag(
                    fuel_system_id=fuel_system_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )
