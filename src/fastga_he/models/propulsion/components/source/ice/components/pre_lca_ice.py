# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAICEProdWeightPerFU


class PreLCAICE(om.Group):
    def initialize(self):
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):
        ice_id = self.options["ice_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAICEProdWeightPerFU(ice_id=ice_id),
            promotes=["*"],
        )
