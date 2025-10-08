# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import logging

_LOGGER = logging.getLogger(__name__)

class SizingToothRatio(om.ExplicitComponent):
    """
    Computation of the ratio between the circumference of the stator bore and the total teeth
    slot length. The formula is obtained from equation (II-48)  in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
            val=np.nan,
            units="T",
            desc="Maximum magnetic flux density provided by the permanent magnets",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max",
            val=np.nan,
            units="T",
            desc="Maximum total magnetic flux density in air gap including electromagnetism",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_flux_density_ratio",
            val=1.4,
            desc="Maximum mean tooth magnetic flux density divided by the maximum air gap flux "
            "density",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            val=0.44,
            desc="The fraction between overall tooth length and stator bore circumference",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        max_total_flux = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max"
        ]
        max_air_gap_flux = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max"
        ]
        tooth_flux_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_flux_density_ratio"
        ]

        if 0 < 2.0 * max_total_flux / (np.pi * max_air_gap_flux * tooth_flux_ratio) < 1.0:
            _LOGGER.info(
                msg="Tooth ratio set to 0.5, please check tooth flux ratio."
            )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"] = (
            2.0 * max_total_flux / (np.pi * max_air_gap_flux * tooth_flux_ratio)
            if (0 < 2.0 * max_total_flux / (np.pi * max_air_gap_flux * tooth_flux_ratio) < 1.0)
            else 0.5
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        max_total_flux = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max"
        ]
        max_air_gap_flux = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max"
        ]
        tooth_flux_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_flux_density_ratio"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max",
        ] = (
            2.0 / (np.pi * max_air_gap_flux * tooth_flux_ratio)
            if (0 < 2.0 * max_total_flux / (np.pi * max_air_gap_flux * tooth_flux_ratio) < 1.0)
            else 0.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
        ] = (
            -2.0 * max_total_flux / (np.pi * max_air_gap_flux**2.0 * tooth_flux_ratio)
            if (0 < 2.0 * max_total_flux / (np.pi * max_air_gap_flux * tooth_flux_ratio) < 1.0)
            else 0.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_flux_density_ratio",
        ] = (
            -2.0 * max_total_flux / (np.pi * max_air_gap_flux * tooth_flux_ratio**2.0)
            if (0 < 2.0 * max_total_flux / (np.pi * max_air_gap_flux * tooth_flux_ratio) < 1.0)
            else 0.5
        )
