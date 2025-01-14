# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCATurboGeneratorProdWeightPerFU


class PreLCATurboGenerator(om.Group):
    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):
        turbo_generator_id = self.options["turbo_generator_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCATurboGeneratorProdWeightPerFU(turbo_generator_id=turbo_generator_id),
            promotes=["*"],
        )
