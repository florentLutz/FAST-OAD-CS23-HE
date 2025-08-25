# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_weight_per_fu import PreLCAMotorProdWeightPerFU


class PreLCAPMSM(om.Group):
    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_subsystem(
            name="weight_per_fu",
            subsys=PreLCAMotorProdWeightPerFU(pmsm_id=pmsm_id),
            promotes=["*"],
        )
