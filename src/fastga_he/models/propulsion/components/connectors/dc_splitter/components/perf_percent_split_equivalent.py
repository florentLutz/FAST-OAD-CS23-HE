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

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="dc_current_out",
            units="A",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Current going out of the bus at the output",
        )
        self.add_input(
            name="dc_voltage",
            units="V",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Voltage of the splitter",
        )
        self.add_input(
            "power_share",
            units="W",
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
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        power_share = inputs["power_share"]
        dc_current_out = inputs["dc_current_out"]
        dc_voltage = inputs["dc_voltage"]

        power_split = np.clip(100.0 * power_share / (dc_current_out * dc_voltage), 0.0, 100.0)

        outputs["power_split"] = power_split

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        power_output = inputs["dc_voltage"] * inputs["dc_current_out"]
        power_share = inputs["power_share"]

        # In case the power output is below the power share, the percent split should always be
        # 100% but that will means that the partials are at zero potentially only during a few
        # point of the mission which the solver might interpret as the Jacobian not being full in
        # rank. Consequently, instead of putting it at 0 we put a very small value.
        partials_power_share = np.where(power_share < power_output, 100.0 / power_output, 0.0)
        partials["power_split", "power_share"] = partials_power_share

        partials_voltage_out = np.where(
            power_share < power_output,
            -100.0 * power_share / power_output / inputs["dc_voltage"],
            1e-6,
        )
        partials["power_split", "dc_voltage"] = partials_voltage_out

        partials_current_out = np.where(
            power_share < power_output,
            -100.0 * power_share / power_output / inputs["dc_current_out"],
            1e-6,
        )
        partials["power_split", "dc_current_out"] = partials_current_out
