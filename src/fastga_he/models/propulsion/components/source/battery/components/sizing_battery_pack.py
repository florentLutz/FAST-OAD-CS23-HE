# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_module_weight import SizingBatteryModuleWeight
from .sizing_battery_weight import SizingBatteryWeight
from .sizing_number_cells import SizingBatteryNumberCells
from .sizing_module_volume import SizingBatteryModuleVolume
from .sizing_battery_volume import SizingBatteryVolume
from .sizing_battery_dimensions import SizingBatteryDimensions
from .sizing_battery_cg_x import SizingBatteryCGX
from .sizing_battery_cg_y import SizingBatteryCGY
from .sizing_battery_drag import SizingBatteryDrag
from .sizing_battery_prep_for_loads import SizingBatteryPreparationForLoads


from .cstr_battery_pack import ConstraintsBattery

from ..constants import POSSIBLE_POSITION


class SizingBatteryPack(om.Group):
    """Class that regroups all of the sub components for the sizing of the battery pack."""

    def initialize(self):

        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the battery, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        battery_pack_id = self.options["battery_pack_id"]
        position = self.options["position"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        self.add_subsystem(
            name="constraints_battery",
            subsys=ConstraintsBattery(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )

        self.add_subsystem(
            name="number_of_cells",
            subsys=SizingBatteryNumberCells(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="module_weight",
            subsys=SizingBatteryModuleWeight(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="module_volume",
            subsys=SizingBatteryModuleVolume(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_weight",
            subsys=SizingBatteryWeight(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_volume",
            subsys=SizingBatteryVolume(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_dimensions",
            subsys=SizingBatteryDimensions(battery_pack_id=battery_pack_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_CG_x",
            subsys=SizingBatteryCGX(battery_pack_id=battery_pack_id, position=position),
            promotes=["*"],
        )
        self.add_subsystem(
            name="battery_CG_y",
            subsys=SizingBatteryCGY(battery_pack_id=battery_pack_id, position=position),
            promotes=["*"],
        )
        for low_speed_aero in [True, False]:
            system_name = "battery_drag_ls" if low_speed_aero else "battery_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingBatteryDrag(
                    battery_pack_id=battery_pack_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )

        if position == "inside_the_wing":

            self.add_subsystem(
                name="preparation_for_loads",
                subsys=SizingBatteryPreparationForLoads(
                    battery_pack_id=battery_pack_id,
                    position=position,
                ),
                promotes=["*"],
            )
