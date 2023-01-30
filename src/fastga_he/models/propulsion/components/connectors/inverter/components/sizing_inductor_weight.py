# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterInductorWeight(om.ExplicitComponent):
    """
    Computation of the weight of the inductor, implementation of the formula from
    :cite:`giraud:2014`.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the inverter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance",
            units="H",
            val=np.nan,
            desc="Inductance of the inductor",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:mass",
            units="kg",
            val=7,
            desc="Mass of the 3 single phase inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]
        inductance = (
            inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance"
            ]
            * 1e6
        )

        inductors_weight = 1.45e-4 * inductance ** 0.85 * current_caliber ** 1.68

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:mass"
        ] = inductors_weight

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]
        inductance = (
            inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance"
            ]
            * 1e6
        )

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = (
            1.68 * 1.45e-4 * inductance ** 0.85 * current_caliber ** 0.68
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance",
        ] = (
            0.85
            * 1.45e-4
            * 1e6 ** 0.85
            * current_caliber ** 1.68
            * inputs[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance"
            ]
            ** -0.15
        )
