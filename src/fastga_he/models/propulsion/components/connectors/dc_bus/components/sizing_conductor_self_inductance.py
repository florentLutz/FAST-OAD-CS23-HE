# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBusBarSelfInductance(om.ExplicitComponent):
    """
    Computation of the conductor plates self inductance, as both are assumed to have the same
    dimensions their self inductance will be the same.

    Based on the formula from :cite:`zhu:2006`.
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length",
            units="m",
            val=0.3,
            desc="Length of the bus bar conductor",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:"
            + dc_bus_id
            + ":conductor:self_inductance",
            units="H",
            val=2e-7,
            desc="Self inductance of the bus bar conductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        conductor_thickness = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness"
        ]
        conductor_width = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width"
        ]
        conductor_length = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length"
        ]

        self_inductance = (
            2e-7
            * conductor_length
            * (
                2.303 * np.log10(2.0 * conductor_length / (conductor_width + conductor_thickness))
                + 0.5
                + 0.2235 * (conductor_width + conductor_thickness) / conductor_length
            )
        )

        outputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:self_inductance"
        ] = self_inductance

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        conductor_thickness = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness"
        ]
        conductor_width = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width"
        ]
        conductor_length = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length"
        ]

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:self_inductance",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length",
        ] = (
            4.606e-7
            * (np.log(2.0 / (conductor_thickness + conductor_width) * conductor_length) + 1)
            / np.log(10)
            + 1e-7
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:self_inductance",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
        ] = 2e-7 * (
            0.2235 - 1.000179082 * conductor_length / (conductor_thickness + conductor_width)
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":conductor:self_inductance",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
        ] = 2e-7 * (
            0.2235 - 1.000179082 * conductor_length / (conductor_thickness + conductor_width)
        )
