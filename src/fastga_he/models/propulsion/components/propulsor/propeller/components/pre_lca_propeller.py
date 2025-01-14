# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAPropellerProdWeightPerFU


class PreLCAPropeller(om.Group):
    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAPropellerProdWeightPerFU(propeller_id=propeller_id),
            promotes=["*"],
        )
