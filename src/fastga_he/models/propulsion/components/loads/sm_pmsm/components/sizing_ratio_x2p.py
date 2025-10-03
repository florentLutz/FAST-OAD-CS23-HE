# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRatioX2p(om.ExplicitComponent):
    """
    Computation of a fraction in the air gap flux density calculation. This ratio consists of the
    radius ratio and the number of pole pairs. The formula is obtained from equation (
    II-24) and (II-25) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pairs in the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
            val=np.nan,
            desc="the radius ratio of the rotor radius and the stator bore radius",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
            val=16.436,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        radius_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio"] = (
            1.0 + radius_ratio ** (2.0 * num_pole_pairs)
        ) / (1.0 - radius_ratio ** (2.0 * num_pole_pairs))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        radius_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
        ] = (
            4.0
            * num_pole_pairs
            * radius_ratio ** (2.0 * num_pole_pairs - 1.0)
            / (radius_ratio ** (2.0 * num_pole_pairs) - 1.0) ** 2.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = (
            4.0
            * radius_ratio ** (2.0 * num_pole_pairs)
            * np.log(radius_ratio)
            / (radius_ratio ** (2.0 * num_pole_pairs) - 1.0) ** 2.0
        )
