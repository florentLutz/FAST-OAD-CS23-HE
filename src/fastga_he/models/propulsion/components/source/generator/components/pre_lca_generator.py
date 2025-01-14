# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAGeneratorProdWeightPerFU


class PreLCAGenerator(om.Group):
    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        generator_id = self.options["generator_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAGeneratorProdWeightPerFU(generator_id=generator_id),
            promotes=["*"],
        )
