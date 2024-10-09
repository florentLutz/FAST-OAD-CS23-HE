# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .pre_lca_prod_length_per_fu import PreLCAHarnessProdLengthPerFU


class PreLCAHarness(om.Group):
    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):
        harness_id = self.options["harness_id"]

        self.add_subsystem(
            name="length_per_fu",
            subsys=PreLCAHarnessProdLengthPerFU(harness_id=harness_id),
            promotes=["*"],
        )
