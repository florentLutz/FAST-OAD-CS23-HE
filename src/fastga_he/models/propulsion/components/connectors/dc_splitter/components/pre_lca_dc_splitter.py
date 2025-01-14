# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCADCSplitterProdWeightPerFU


class PreLCADCSplitter(om.Group):
    def initialize(self):
        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            allow_none=False,
        )

    def setup(self):
        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCADCSplitterProdWeightPerFU(dc_splitter_id=dc_splitter_id),
            promotes=["*"],
        )
