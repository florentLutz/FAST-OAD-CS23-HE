# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHeatSinkTubeOuterDiameter(om.ExplicitComponent):
    """
    Computation of outer diameter of the tube running in the heat sink based on the inner
    diameter and thickness. Method from :cite:`giraud:2014`.
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
            name=prefix + ":heat_sink:tube:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the tube for the cooling of the inverter",
        )
        self.add_input(
            name=prefix + ":heat_sink:tube:thickness",
            units="m",
            val=1.25e-3,
            desc="Thickness of the tube for the cooling of the inverter",
        )

        self.add_output(
            name=prefix + ":heat_sink:tube:outer_diameter",
            units="m",
            val=0.0025,
            desc="Outer diameter of the tube for the cooling of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        outputs[prefix + ":heat_sink:tube:outer_diameter"] = (
            inputs[prefix + ":heat_sink:tube:inner_diameter"]
            + 2.0 * inputs[prefix + ":heat_sink:tube:thickness"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        partials[
            prefix + ":heat_sink:tube:outer_diameter", prefix + ":heat_sink:tube:inner_diameter"
        ] = 1.0
        partials[
            prefix + ":heat_sink:tube:outer_diameter", prefix + ":heat_sink:tube:thickness"
        ] = 2.0
