# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCLandingGearCostReduction(om.ExplicitComponent):
    """
    Computation of the landing gear cost reduction per aircraft from :cite:`gudmundsson:2013`.
    The cost of retractable landing gear is already include in other cost. Thus, the cost per
    aircraft needs to be adjusted as the fixed design is considered.
    """

    def initialize(self):
        self.options.declare(
            name="retractable_landing_gear",
            default=False,
            types=bool,
            desc="True if retractable design is selected",
        )

    def setup(self):
        if self.options["retractable_landing_gear"]:
            self.add_output(
                "data:cost:airframe:landing_gear_cost_reduction_per_unit",
                val=0.0,
                units="USD",
                desc="Cost reduction if fixed landing gear design is selected",
            )
        else:
            self.add_output(
                "data:cost:airframe:landing_gear_cost_reduction_per_unit",
                val=-7500.0,
                units="USD",
                desc="Cost reduction if fixed landing gear design is selected",
            )
