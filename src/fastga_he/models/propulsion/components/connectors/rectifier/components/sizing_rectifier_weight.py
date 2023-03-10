# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRectifierWeight(om.ExplicitComponent):
    """
    Computation of the rectifier weight, based on power density. Default value of power density
    is based on the 2025 target in :cite:`pettes:2021.`
    """

    def initialize(self):

        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            types=str,
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            val=np.nan,
            units="A",
        )
        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
            val=np.nan,
            units="V",
        )
        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density",
            val=15.0,
            units="kW/kg",
        )

        self.add_output(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            val=15.0,
            units="kg",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]

        current_rms_1_phase = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]
        voltage_rms = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
        ] / np.sqrt(2.0)
        power_density = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density"
        ]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass"] = (
            3.0 * current_rms_1_phase * voltage_rms / (power_density * 1000.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rectifier_id = self.options["rectifier_id"]

        current_rms_1_phase = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]
        voltage_rms = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
        ] / np.sqrt(2.0)
        power_density = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density"
        ]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = (
            3.0 * voltage_rms / (power_density * 1000.0)
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
        ] = (
            3.0 * current_rms_1_phase / (power_density * np.sqrt(2.0) * 1000.0)
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":mass",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":power_density",
        ] = -(3.0 * current_rms_1_phase * voltage_rms / (1000.0 * power_density ** 2.0))
