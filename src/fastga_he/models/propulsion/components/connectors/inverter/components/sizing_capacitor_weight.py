# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCapacitorWeight(om.ExplicitComponent):
    """
    Computation of the weight of the capacitor, regression based on capacitors from AVX,
    does not take rms current into account but it is an important criterion when choosing data
    from a catalog.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity",
            val=np.nan,
            units="F",
            desc="Capacity required to dampen the design voltage ripple",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:mass",
            val=5000.0,
            units="g",
            desc="Mass of the capacitor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        # We need the capacity in microF
        capacity = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity"]
            * 1e6
        )

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:mass"] = (
            10.524 * capacity ** 0.7749
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity",
        ] = (
            10.524
            * 1e6 ** 0.7749
            * 0.7749
            * inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity"
            ]
            ** (0.7749 - 1.0)
        )
