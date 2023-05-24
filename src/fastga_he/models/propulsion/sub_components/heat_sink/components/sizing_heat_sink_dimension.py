# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHeatSinkDimension(om.ExplicitComponent):
    """
    Computation of the dimension of the heat sink of the component (plaque froide in french).
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
            name=prefix + ":module:length",
            units="m",
            val=np.nan,
            desc="Length of one module",
        )
        self.add_input(
            name=prefix + ":module:width",
            units="m",
            val=np.nan,
            desc="Width of one module",
        )

        self.add_output(
            name=prefix + ":heat_sink:length",
            units="m",
            val=0.20,
            desc="Length of the heat sink",
        )
        self.add_output(
            name=prefix + ":heat_sink:width",
            units="m",
            val=0.17,
            desc="Width of the heat sink",
        )

        self.declare_partials(
            of=prefix + ":heat_sink:length",
            wrt=prefix + ":module:width",
            method="exact",
        )
        self.declare_partials(
            of=prefix + ":heat_sink:width",
            wrt=prefix + ":module:length",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":heat_sink:length"] = 3.3 * inputs[prefix + ":module:width"]
        outputs[prefix + ":heat_sink:width"] = 1.1 * inputs[prefix + ":module:length"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        prefix = self.options["prefix"]

        partials[prefix + ":heat_sink:length", prefix + ":module:width"] = 3.3
        partials[prefix + ":heat_sink:width", prefix + ":module:length"] = 1.1
