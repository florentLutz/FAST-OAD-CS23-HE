# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualEnergyCost(om.ExplicitComponent):
    """
    Computation of the yearly energy cost of the aircraft, the sum of the fuel and the electricity
    cost.
    """

    def setup(self):
        self.add_input(
            name="data:operation:annual_fuel_cost",
            val=np.nan,
            units="USD/yr",
        )

        self.add_input(
            name="data:operation:annual_electricity_cost",
            val=np.nan,
            units="USD/yr",
        )

        self.add_output(
            name="data:operation:annual_energy_cost",
            val=2000.0,
            units="USD/yr",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:operation:annual_energy_cost"] = np.sum(inputs.values())
