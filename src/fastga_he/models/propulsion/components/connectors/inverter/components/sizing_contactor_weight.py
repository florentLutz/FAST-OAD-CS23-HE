# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterContactorWeight(om.ExplicitComponent):
    """
    Computation of the weight of the contactors, based on a regression performed on the data from
    :cite:`giraud:2014`. Assumes that there are 3 contactors in the inverter. Correlation can be
    found in ..methodology.contactor_mass.py.
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

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":contactor:mass",
            units="kg",
            val=4.5,
            desc="Mass of the 3 contactors",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":contactor:mass"] = (
            3.0 * 0.0239 * current_caliber ** 0.6942
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":contactor:mass",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = (
            3.0 * 0.0239 * 0.6942 * current_caliber ** (0.6942 - 1.0)
        )
