# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import DEFAULT_MAX_CURRENT_DENSITY


class PerformancesPhaseCurrentDensity(om.ExplicitComponent):
    """
    Computation of the phase current density of the SM PMSM. The formula is obtained from
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
            "ac_current_rms_in_one_phase",
            units="kA",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_current_density",
            units="kA/m**2",
            val=DEFAULT_MAX_CURRENT_DENSITY,
            desc="Upper limit of the current density for air cooling PMSM ",
        )

        self.add_output(
            "ac_phase_current_density",
            units="kA/m**2",
            val=3.0e3,
            shape=number_of_points,
            desc="The phase current density of the SM PMSM",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="ac_current_rms_in_one_phase",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":wire_circular_section_area",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_current_density",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["ac_phase_current_density"] = np.clip(
            inputs["ac_current_rms_in_one_phase"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ],
            0.0,
            inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_current_density"
            ],
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]
        design_current_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_current_density"
        ]
        unclipped_current_density = (
            inputs["ac_current_rms_in_one_phase"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ]
        )
        clipped_current_density = np.clip(unclipped_current_density, 1.0e-3, design_current_density)

        partials[
            "ac_phase_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area",
        ] = np.where(
            unclipped_current_density == clipped_current_density,
            -inputs["ac_current_rms_in_one_phase"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ]
            ** 2.0,
            1.0e-6,
        )

        partials["ac_phase_current_density", "ac_current_rms_in_one_phase"] = np.where(
            unclipped_current_density == clipped_current_density,
            np.ones(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ],
            1.0e-6,
        )

        partials[
            "ac_phase_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_current_density",
        ] = np.where(
            unclipped_current_density == clipped_current_density,
            np.zeros(number_of_points),
            1.0,
        )
