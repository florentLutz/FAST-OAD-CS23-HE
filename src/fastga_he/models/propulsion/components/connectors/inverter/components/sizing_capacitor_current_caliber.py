# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCapacitorCurrentCaliber(om.ExplicitComponent):
    """
    Computation of the maximum current that can pass through the capacitor. Instead of taking the
    maximum of the different cases as in :cite:`giraud:2014`, we take the worst case scenario.
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
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_factor",
            val=1.0,
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":capacitor:current_caliber",
            units="A",
            val=200.0,
            desc="Current caliber of the capacitor of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        cos_phi = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":power_factor"]

        # Maximum of the term that multiplies the current caliber of the modules is found thanks
        # to the following formula obtained by a simple derivative computation

        factor = np.sqrt(
            2.0 / (3.0 * np.pi ** 2.0) * (1 + 2.0 * cos_phi ** 2.0 + 1.0 / (8.0 * cos_phi ** 2.0))
        )

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:current_caliber"
        ] = (
            factor
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        cos_phi = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":power_factor"]

        factor = np.sqrt(
            2.0 / (3.0 * np.pi ** 2.0) * (1 + 2.0 * cos_phi ** 2.0 + 1.0 / (8.0 * cos_phi ** 2.0))
        )

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:current_caliber",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = factor
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:current_caliber",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_factor",
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
            / (2.0 * factor)
            * 2.0
            / (3.0 * np.pi ** 2.0)
            * (4.0 * cos_phi - 1.0 / (4.0 * cos_phi ** 3.0))
        )
