# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesRMSLinearCurrentDensity(om.ExplicitComponent):
    """
    Computation of the linear current density of the SM PMSM. The formula is obtained from
    equation (II-38) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_density_rms",
            units="A/m**2",
            val=np.nan,
            shape=number_of_points,
            desc="The RMS current density of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor",
            val=np.nan,
            desc="The area factor considers the cross-section shape twist due to wire bunching",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
            val=np.nan,
            desc="The factor describes the conductor material fullness inside the stator slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            units="m",
            desc="Single stator slot height (radial)",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            val=np.nan,
            desc="The fraction between overall tooth length and stator bore circumference",
        )

        self.add_output(
            "ac_linear_current_density_rms",
            units="A/m",
            val=0.0,
            shape=number_of_points,
            desc="The linear RMS current density of the SM PMSM",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="ac_linear_current_density_rms",
            wrt="ac_current_density_rms",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="ac_linear_current_density_rms",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        current_density_rms = inputs["ac_current_density_rms"]
        slot_conductor_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor"
        ]
        slot_fill_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"
        ]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]

        outputs["ac_linear_current_density_rms"] = (
            slot_conductor_factor
            * slot_fill_factor
            * slot_height
            * current_density_rms
            * (1.0 - tooth_ratio)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        current_density_rms = inputs["ac_current_density_rms"]
        slot_conductor_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor"
        ]
        slot_fill_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"
        ]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]

        partials[
            "ac_linear_current_density_rms",
            "ac_current_density_rms",
        ] = np.full(
            number_of_points,
            slot_conductor_factor * slot_fill_factor * slot_height * (1.0 - tooth_ratio),
        )

        partials[
            "ac_linear_current_density_rms",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor",
        ] = slot_fill_factor * slot_height * current_density_rms * (1.0 - tooth_ratio)

        partials[
            "ac_linear_current_density_rms",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
        ] = slot_conductor_factor * slot_height * current_density_rms * (1.0 - tooth_ratio)

        partials[
            "ac_linear_current_density_rms",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
        ] = slot_conductor_factor * slot_fill_factor * current_density_rms * (1.0 - tooth_ratio)

        partials[
            "ac_linear_current_density_rms",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
        ] = -slot_conductor_factor * slot_fill_factor * slot_height * current_density_rms
