# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCARectifierProdWeightPerFU


class PreLCARectifier(om.Group):
    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            types=str,
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCARectifierProdWeightPerFU(rectifier_id=rectifier_id),
            promotes=["*"],
        )