# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCABatteryProdWeightPerFU
from .pre_lca_use_emission_per_fu import PreLCABatteryUseEmissionPerFU


class PreLCABatteryPack(om.Group):
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
        self.add_subsystem(
            name="emissions_per_fu",
            subsys=PreLCABatteryUseEmissionPerFU(battery_pack_id=battery_pack_id),
            promotes=["*"],
        )