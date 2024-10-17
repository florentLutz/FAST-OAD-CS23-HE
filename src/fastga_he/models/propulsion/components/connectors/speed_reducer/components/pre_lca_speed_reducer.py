# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCASpeedReducerProdWeightPerFU


class PreLCASpeedReducer(om.Group):
    def initialize(self):
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )

    def setup(self):
        speed_reducer_id = self.options["speed_reducer_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCASpeedReducerProdWeightPerFU(speed_reducer_id=speed_reducer_id),
            promotes=["*"],
        )
