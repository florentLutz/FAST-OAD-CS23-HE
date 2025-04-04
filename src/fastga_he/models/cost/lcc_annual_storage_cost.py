# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class LCCAnnualStorageCost(om.ExplicitComponent):
    """
    Computation of the annual storage cost, obtained by averaging fee from several hangers.
    """

    def setup(self):
        self.add_input(
            "data:cost:storage_fee_per_month",
            units="USD",
            val=600.0,
            desc="Mean of the monthly storage fee across several hangers",
        )

        self.add_output(
            "data:cost:annual_storage_cost",
            val=1500.0,
            units="USD/yr",
            desc="Annual storage cost of the aircraft",
        )
        self.declare_partials("*", "*", val=12.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:annual_storage_cost"] = 12.0 * inputs["data:cost:storage_fee_per_month"]
