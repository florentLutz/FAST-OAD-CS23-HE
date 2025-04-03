# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class LCCFuelTankCost(om.ExplicitComponent):
    """
    Computation of fuel tank cost based on the fuel tank purchase price of Cessna 150.
    site: https://www.preferredairparts.com/new-surplus-cessna-150-right-hand-fuel-tank-pn-0426508-22
    """

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
            units="galUS",
            val=np.nan,
            desc="Capacity of the tank in term of volume",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":cost_per_tank",
            units="USD",
            val=4.0e3,
            desc="Purchase cost per fuel tank",
        )

        self.declare_partials(of="*", wrt="*", val=40.4)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]

        outputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":cost_per_tank"] = (
            40.4 * inputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"]
        )
