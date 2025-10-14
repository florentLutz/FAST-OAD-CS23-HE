# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCurrentDensityRMS(om.ExplicitComponent):
    """
    Computation of the RMS current density of the SM PMSM. The formula is obtained from
    equation (II-32) in :cite:`touhami:2020`.
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
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area",
            units="m**2",
            val=np.nan,
            desc="Single conductor circular wire cross-section area",
        )
        self.add_input(
            "ac_current_rms_in",
            units="A",
            val=np.nan,
            shape=number_of_points,
            desc="RMS current in all the phases of the motor",
        )

        self.add_output(
            "ac_current_density_rms",
            units="A/m**2",
            val=1.0e4,
            shape=number_of_points,
            desc="The current density of the SM PMSM",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="ac_current_density_rms",
            wrt="ac_current_rms_in",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="ac_current_density_rms",
            wrt="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":wire_circular_section_area",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["ac_current_density_rms"] = (
            inputs["ac_current_rms_in"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        partials[
            "ac_current_density_rms",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area",
        ] = (
            inputs["ac_current_rms_in"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ]
            ** 2.0
        )

        partials["ac_current_density_rms", "ac_current_rms_in"] = (
            np.ones(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ]
        )
