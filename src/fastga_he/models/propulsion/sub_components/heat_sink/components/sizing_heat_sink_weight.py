# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHeatSinkWeight(om.ExplicitComponent):
    """
    Computation of the weight of the heat sink, computed as the sum of the weight of the tube and of
    the "core" which is assumed to be made out of aluminium. Fluid weight is neglected. Method from
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
            name=prefix + ":heat_sink:tube:mass",
            units="kg",
            val=np.nan,
            desc="Outer diameter of the tube for the cooling of the inverter",
        )
        self.add_input(
            name=prefix + ":heat_sink:tube:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the tube for the cooling of the inverter",
        )
        self.add_input(
            name=prefix + ":heat_sink:length",
            units="m",
            val=np.nan,
            desc="Length of the heat sink",
        )
        self.add_input(
            name=prefix + ":heat_sink:width",
            units="m",
            val=np.nan,
            desc="Width of the heat sink",
        )
        self.add_input(
            name=prefix + ":heat_sink:height",
            units="m",
            val=np.nan,
            desc="Height of the heat sink",
        )
        self.add_input(
            name=prefix + ":tube:number_of_passes",
            val=4,
            desc="Number of passes in the heat sink (between 2 and 6 usually)",
        )
        self.add_input(
            name=prefix + ":heat_sink:density",
            units="kg/m**3",
            val=2700.0,
            desc="Density of the heat sink core for the inverter, by default aluminium is assumed",
        )

        self.add_output(
            name=prefix + ":heat_sink:mass",
            units="kg",
            val=1.0,
            desc="Mass of the heat sink, includes tubes and core",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        tube_weight = inputs[prefix + ":heat_sink:tube:mass"]
        length_hs = inputs[prefix + ":heat_sink:length"]
        width_hs = inputs[prefix + ":heat_sink:width"]
        height_hs = inputs[prefix + ":heat_sink:height"]
        number_of_passes = inputs[prefix + ":tube:number_of_passes"]
        outer_diameter = inputs[prefix + ":heat_sink:tube:outer_diameter"]
        core_density = inputs[prefix + ":heat_sink:density"]

        weight_hs = (
            tube_weight
            + (
                height_hs * length_hs * width_hs
                - np.pi / 4 * number_of_passes * length_hs * outer_diameter ** 2.0
            )
            * core_density
        )

        outputs[prefix + ":heat_sink:mass"] = weight_hs

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        length_hs = inputs[prefix + ":heat_sink:length"]
        width_hs = inputs[prefix + ":heat_sink:width"]
        height_hs = inputs[prefix + ":heat_sink:height"]
        number_of_passes = inputs[prefix + ":tube:number_of_passes"]
        outer_diameter = inputs[prefix + ":heat_sink:tube:outer_diameter"]
        core_density = inputs[prefix + ":heat_sink:density"]

        partials[
            prefix + ":heat_sink:mass",
            prefix + ":heat_sink:tube:mass",
        ] = 1.0
        partials[prefix + ":heat_sink:mass", prefix + ":heat_sink:length",] = (
            height_hs * width_hs - np.pi / 4 * number_of_passes * outer_diameter ** 2.0
        ) * core_density
        partials[prefix + ":heat_sink:mass", prefix + ":heat_sink:width",] = (
            height_hs * length_hs * core_density
        )
        partials[prefix + ":heat_sink:mass", prefix + ":heat_sink:height",] = (
            width_hs * length_hs * core_density
        )
        partials[
            prefix + ":heat_sink:mass",
            prefix + ":tube:number_of_passes",
        ] = (-np.pi / 4 * length_hs * outer_diameter ** 2.0) * core_density
        partials[prefix + ":heat_sink:mass", prefix + ":heat_sink:tube:outer_diameter",] = (
            -np.pi / 2 * number_of_passes * length_hs * outer_diameter * core_density
        )
        partials[prefix + ":heat_sink:mass", prefix + ":heat_sink:density",] = (
            height_hs * length_hs * width_hs
            - np.pi / 4 * number_of_passes * length_hs * outer_diameter ** 2.0
        )
