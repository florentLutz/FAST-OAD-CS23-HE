# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRectifierEnergyCoefficientScaling(om.ExplicitComponent):
    """Computation of scaling ratio for the loss coefficient of the rectifier."""

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

        self.options.declare(
            name="current_caliber_ref",
            types=float,
            default=450.0,
            desc="Current caliber of the reference component",
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the rectifier",
        )

        self.add_output("data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:a")
        self.add_output("data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:c")

        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:a",
            wrt="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:c",
            wrt="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]

        current_caliber_ref = self.options["current_caliber_ref"]
        current_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]

        current_caliber_star = current_ac_caliber / current_caliber_ref

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:a"
        ] = current_caliber_star
        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:c"] = (
            current_caliber_star ** -1
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]
        current_caliber_ref = self.options["current_caliber_ref"]
        current_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:a",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = (
            1.0 / current_caliber_ref
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:c",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = (
            -current_caliber_ref / current_ac_caliber ** 2.0
        )
