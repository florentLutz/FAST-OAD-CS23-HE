# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterHeatSinkTubeOuterDiameter(om.ExplicitComponent):
    """
    Computation of outer diameter of the tube running in the heat sink based on the inner
    diameter and thickness. Method from :cite:`giraud:2014`.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the tube for the cooling of the inverter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:thickness",
            units="m",
            val=1.25e-3,
            desc="Thickness of the tube for the cooling of the inverter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter",
            units="m",
            val=0.0025,
            desc="Outer diameter of the tube for the cooling of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter"
        ] = (
            inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink:tube:inner_diameter"
            ]
            + 2.0
            * inputs[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":heat_sink:tube:thickness"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:inner_diameter",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":heat_sink:tube:outer_diameter",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":heat_sink:tube:thickness",
        ] = 2.0
