# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPercentSplitEquivalent(om.ExplicitComponent):
    """
    The power split mode has been extensively tested and seem to work even in strange cases
    so to implement the power share mode, we will convert the power share in a power split.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        planetary_gear_id = self.options["planetary_gear_id"]

        self.add_input(
            name="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":efficiency",
            val=0.98,
            desc="Efficiency of the planetary gear",
        )
        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points)
        self.add_input(
            "power_share",
            units="kW",
            val=np.nan,
            shape=number_of_points,
            desc="Share of the power going to the first (primary) input, in W, with a format "
            "adapted to mission. If below nothing will go in the secondary input, if above, "
            "the complement will flow in the secondary input",
        )

        self.add_output(
            "power_split",
            units="percent",
            val=50.0,
            shape=number_of_points,
            desc="Percent of the power going to the first (primary) input, in %, with a format "
            "adapted to mission",
        )

        self.declare_partials(
            of="power_split",
            wrt=["power_share", "shaft_power_out"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="power_split",
            wrt="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":efficiency",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]

        power_share = inputs["power_share"]
        power_out = inputs["shaft_power_out"]
        eta = inputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":efficiency"
        ]

        power_split = np.clip(100.0 * power_share / (power_out / eta), 0.0, 100.0)

        outputs["power_split"] = power_split

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]

        power_share = inputs["power_share"]
        power_out = inputs["shaft_power_out"]
        eta = inputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":efficiency"
        ]

        power_in = power_out / eta

        # In case the power output is below the power share, the percent split should always be
        # 100% but that will means that the partials are at zero potentially only during a few
        # point of the mission which the solver might interpret as the Jacobian not being full in
        # rank. Consequently, instead of putting it at 0 we put a very small value.
        partials_power_share = np.where(power_share < power_in, 100.0 / power_in, 1e-6)
        partials["power_split", "power_share"] = partials_power_share

        partials["power_split", "shaft_power_out"] = np.where(
            power_share < power_in, -100.0 * power_share / (power_out ** 2.0 / eta), -1e-6
        )
        partials[
            "power_split",
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":efficiency",
        ] = np.where(power_share < power_in, 100.0 * power_share / power_out, 1e-6)
