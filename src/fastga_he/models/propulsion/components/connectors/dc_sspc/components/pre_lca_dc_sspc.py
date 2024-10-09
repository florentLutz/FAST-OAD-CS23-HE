# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCADCSSPCProdWeightPerFU


class PreLCADCSSPC(om.Group):
    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCADCSSPCProdWeightPerFU(dc_sspc_id=dc_sspc_id),
            promotes=["*"],
        )
