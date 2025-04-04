# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCCertificationCost(om.ExplicitComponent):
    """
    Computation of the certification cost per aircraft from :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input(
            "data:cost:engineering_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Engineering adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:dev_support_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Development support adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:flight_test_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Development flight test adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:tooling_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Tooling adjusted cost per aircraft",
        )

        self.add_output(
            "data:cost:certification_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Certification adjusted cost per aircraft",
        )
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:certification_cost_per_unit"] = np.sum(inputs.values())
