#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCADCAuxLoadProdWeightPerFU


class PreLCADCAuxLoad(om.Group):
    def initialize(self):
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):
        aux_load_id = self.options["aux_load_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCADCAuxLoadProdWeightPerFU(aux_load_id=aux_load_id),
            promotes=["*"],
        )
