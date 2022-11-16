# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_module_weight import SizingBatteryModuleWeight
from .sizing_battery_weight import SizingBatteryWeight


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
