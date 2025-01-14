# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAPlanetaryGearProdWeightPerFU


class PreLCAPlanetaryGear(om.Group):
    def initialize(self):
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )

    def setup(self):
        planetary_gear_id = self.options["planetary_gear_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAPlanetaryGearProdWeightPerFU(planetary_gear_id=planetary_gear_id),
            promotes=["*"],
        )
