# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAInverterProdWeightPerFU


class PreLCAInverter(om.Group):
    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):
        inverter_id = self.options["inverter_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAInverterProdWeightPerFU(inverter_id=inverter_id),
            promotes=["*"],
        )
