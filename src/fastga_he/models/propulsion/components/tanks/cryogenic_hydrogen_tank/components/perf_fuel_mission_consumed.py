# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesLiquidHydrogenConsumedMission(om.ExplicitComponent):
    """
    Computation of the amount of hydrogen in that particular tank which will be consumed. Simple
    summation.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_input(
            "fuel_consumed_t",
            units="kg",
            val=np.full(number_of_points, np.nan),
            desc="Hydrogen from this tank consumed at each time step",
        )

        self.add_output(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":fuel_consumed_mission",
            units="kg",
            val=21.525,
            desc="Amount of fuel from that tank which will be consumed during mission",
        )

        self.declare_partials(of="*", wrt="*", val=np.ones(number_of_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        outputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":fuel_consumed_mission"
        ] = sum(inputs["fuel_consumed_t"])
