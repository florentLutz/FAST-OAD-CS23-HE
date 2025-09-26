# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirGapFluxDensity(om.ExplicitComponent):
    """
    Computation of the air gap magnetic flux density of the mototr. The formula is obtained from
    equation (II-6) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress",
            units="Pa",
            desc="The length of electromagnetism active part of SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density",
            val=np.nan,
            units="A/m",
            desc="The surface current density of the winding conductor cable",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density",
            val=1.0,
            units="T",
            desc="The magnetic flux density provided by the permanent magnets",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density"] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density"
            ]
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density",
        ] = (
            2.0
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress"]
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress",
        ] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density"
            ]
        )
