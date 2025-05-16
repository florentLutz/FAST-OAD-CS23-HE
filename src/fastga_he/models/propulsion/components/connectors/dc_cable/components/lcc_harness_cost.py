# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .lcc_harness_core_unit_cost import LCCHarnessCoreUnitCost
from .lcc_harness_unit_cost import LCCHarnessUnitCost


class LCCHarnessCost(om.Group):
    """
    Class collect all required computations of the DC cable harness.
    """

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
            name="core_material_cost",
            subsys=LCCHarnessCoreUnitCost(harness_id=harness_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="harness_unit_cost",
            subsys=LCCHarnessUnitCost(harness_id=harness_id),
            promotes=["*"],
        )
