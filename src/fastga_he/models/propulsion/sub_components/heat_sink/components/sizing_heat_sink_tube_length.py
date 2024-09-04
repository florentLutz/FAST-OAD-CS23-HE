# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHeatSinkTubeLength(om.ExplicitComponent):
    """
    Computation of the heat sink tube useful length. Implementation of the formula from
    :cite:`giraud:2014`.
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
            name=prefix + ":heat_sink:length",
            units="m",
            val=np.nan,
            desc="Length of the heat sink",
        )
        self.add_input(
            name=prefix + ":tube:number_of_passes",
            val=4,
            desc="Number of passes in the heat sink (between 2 and 6 usually)",
        )

        self.add_output(
            name=prefix + ":heat_sink:tube:length",
            units="m",
            val=0.20,
            desc="Length of the tube which is useful for the cooling of the inverter",
        )

        self.declare_partials(
            of=prefix + ":heat_sink:tube:length",
            wrt=[
                prefix + ":heat_sink:length",
                prefix + ":tube:number_of_passes",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        outputs[prefix + ":heat_sink:tube:length"] = (
            inputs[prefix + ":heat_sink:length"] * inputs[prefix + ":tube:number_of_passes"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        partials[prefix + ":heat_sink:tube:length", prefix + ":heat_sink:length"] = inputs[
            prefix + ":tube:number_of_passes"
        ]
        partials[prefix + ":heat_sink:tube:length", prefix + ":tube:number_of_passes"] = inputs[
            prefix + ":heat_sink:length"
        ]
