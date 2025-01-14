# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAGearboxProdWeightPerFU


class PreLCAGearbox(om.Group):
    def initialize(self):
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )

    def setup(self):
        gearbox_id = self.options["gearbox_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAGearboxProdWeightPerFU(gearbox_id=gearbox_id),
            promotes=["*"],
        )
