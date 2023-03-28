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

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        power_share = inputs["power_share"]
        dc_current_out = inputs["dc_current_out"]
        dc_voltage = inputs["dc_voltage"]

        power_split = np.clip(100.0 * power_share / (dc_current_out * dc_voltage), 0.0, 100.0)

        outputs["power_split"] = power_split

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # The 1e3 is indeed a very unexpected operation here, but without it, for reasons I don't
        # have time to investigate now not having it makes the code not converge
        # TODO: Add it as an option/input and rename it as a "convergence" parameter to make it
        #  sound more legitimate like the compressor map parameters in GasTurb

        power_output = inputs["dc_voltage"] * 1e3 * inputs["dc_current_out"]
        power_share = inputs["power_share"]

        partials["power_split", "power_share"] = np.diag(100.0 / power_output)
        partials["power_split", "dc_voltage"] = np.diag(
            -100.0 * power_share / power_output / inputs["dc_voltage"]
        )
        partials["power_split", "dc_current_out"] = np.diag(
            -100.0 * power_share / power_output / inputs["dc_current_out"],
        )
