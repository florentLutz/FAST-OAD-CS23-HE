# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAH2FuelSystemProdWeightPerFU


class PreLCAH2FuelSystem(om.Group):
    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAH2FuelSystemProdWeightPerFU(h2_fuel_system_id=h2_fuel_system_id),
            promotes=["*"],
        )
