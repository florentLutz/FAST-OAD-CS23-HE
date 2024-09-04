# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHeatSinkHeight(om.ExplicitComponent):
    """
    Computation of the height of the heat sink of the inverter (plaque froide in french).
    Implementation of the formula from :cite:`giraud:2014`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a heatsink",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":heat_sink:tube:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the tube for the cooling of the inverter",
        )

        self.add_output(
            name=prefix + ":heat_sink:height",
            units="m",
            val=0.20,
            desc="Height of the heat sink",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        outputs[prefix + ":heat_sink:height"] = (
            1.5 * inputs[prefix + ":heat_sink:tube:outer_diameter"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        partials[prefix + ":heat_sink:height", prefix + ":heat_sink:tube:outer_diameter"] = 1.5
