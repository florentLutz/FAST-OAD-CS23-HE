# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesSurfaceCurrentDensity(om.ExplicitComponent):
    """
    Computation of the surface current density from the conductors inside the stator slots. The
    formula is obtained from equation (II-35) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor",
            val=np.nan,
            desc="The factor considers the cable winding effect in the stator slot",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":linear_current_density_ac_max",
            units="A/m",
            val=np.nan,
            desc="Maximum value of the RMS linear current density flowing through the motor",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density",
            val=1.0,
            units="A/m",
            desc="The surface current density of the winding conductor cable",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density"
        ] = (
            np.sqrt(2.0)
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":linear_current_density_ac_max"
            ]
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":linear_current_density_ac_max",
        ] = (
            np.sqrt(2.0)
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor"]
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor",
        ] = (
            np.sqrt(2.0)
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":linear_current_density_ac_max"
            ]
        )
