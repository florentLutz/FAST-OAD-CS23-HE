# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAFuelSystemProdWeightPerFU


class PreLCAFuelSystem(om.Group):
    def initialize(self):
        self.options.declare(
            name="fuel_system_id",
            default=None,
            desc="Identifier of the fuel system",
            types=str,
            allow_none=False,
        )

    def setup(self):
        fuel_system_id = self.options["fuel_system_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAFuelSystemProdWeightPerFU(fuel_system_id=fuel_system_id),
            promotes=["*"],
        )
