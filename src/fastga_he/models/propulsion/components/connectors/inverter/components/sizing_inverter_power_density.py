# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterPowerDensity(om.ExplicitComponent):
    """
    Computation of the power density of the inverter, not used in any computation, just there as
    a figure of merit.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the inverter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the inverter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            units="V",
            val=np.nan,
            desc="Voltage caliber of the inverter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":power_density",
            units="kW/kg",
            val=np.nan,
            desc="Power density of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":power_density"] = (
            3.0
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"]
        ) / (1000.0 * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_density",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = (
            3.0
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"]
            / (1000.0 * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"])
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_density",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
        ] = (
            3.0
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
            / (1000.0 * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"])
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":power_density",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
        ] = -(
            3.0
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"]
        ) / (
            1000.0
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"] ** 2.0
        )
