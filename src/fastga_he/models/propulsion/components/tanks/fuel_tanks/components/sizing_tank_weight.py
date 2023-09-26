# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingFuelTankWeight(om.ExplicitComponent):
    """
    Computation of the weight of the tank. The very simplistic approach we will use is to say
    that weight of tank is the weight of unused fuel. Usually the tank weight is included in the
    fuel systems as well as pipes, pumps, ... Since we can't reuse those formula, and since we
    don't have an analytical formula to compute it we will use this simplified approach.
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
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":unusable_fuel_mission",
            units="kg",
            val=np.nan,
            desc="Amount of trapped fuel in the tank",
        )

        self.add_output(
            name="data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass",
            units="kg",
            val=1.0,
            desc="Weight of the fuel tanks",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]

        outputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":mass"] = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":unusable_fuel_mission"
        ]
