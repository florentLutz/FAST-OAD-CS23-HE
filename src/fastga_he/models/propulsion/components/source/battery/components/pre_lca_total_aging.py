#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryTotalAging(om.ExplicitComponent):
    """
    Class that computes the total capacity loss due to aging after x cycles (sum of cyclic and
    calendar aging).
    """

    def setup(self):
        self.add_input(
            name="capacity_loss_cyclic",
            val=np.nan,
            units="unitless",
            desc="Capacity lost due to cyclic aging",
        )
        self.add_input(
            name="capacity_loss_calendar",
            val=0.0,
            units="unitless",
            desc="Capacity lost due to calendar aging",
        )

        self.add_output(
            name="capacity_loss_total",
            val=0.2,
            units="unitless",
            desc="Capacity lost due to total aging",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["capacity_loss_total"] = (
            inputs["capacity_loss_cyclic"] + inputs["capacity_loss_calendar"]
        )
