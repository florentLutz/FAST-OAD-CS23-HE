# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .sizing_module_weight import SizingBatteryModuleWeight
from .sizing_battery_weight import SizingBatteryWeight
from .sizing_number_cells import SizingBatteryNumberCells

from ..constants import SUBMODEL_CONSTRAINTS_BATTERY


class SizingBatteryPack(om.Group):
    """Class that regroups all of the sub components for the sizing of the battery pack."""

    def initialize(self):

        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        battery_pack_id = self.options["battery_pack_id"]

        # It was decided to add the constraints computation at the beginning of the sizing to
        # ensure that both are ran along and to avoid having an additional id to add in the
        # configuration file.
        option_battery_pack_id = {"battery_pack_id": battery_pack_id}

        self.add_subsystem(
            name="constraints_battery",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_BATTERY, options=option_battery_pack_id
            ),
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
            name="battery_weight",
            subsys=SizingBatteryWeight(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
