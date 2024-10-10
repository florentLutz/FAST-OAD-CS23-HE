# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCABatteryProdWeightPerFU


class PreLCABattery(om.Group):
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
            name="weight_per_fu",
            subsys=PreLCABatteryProdWeightPerFU(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )
