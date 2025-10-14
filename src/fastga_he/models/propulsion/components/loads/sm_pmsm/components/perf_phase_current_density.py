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
        self.options.declare(
            "design_current_density",
            default=DEFAULT_MAX_CURRENT_DENSITY,
            desc="Maximum current density [kA/m^2]",
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
            wrt="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":wire_circular_section_area",
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
            DEFAULT_MAX_CURRENT_DENSITY,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]
        unclipped_current_density = (
            inputs["ac_current_rms_in_one_phase"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
            ]
        )
        clipped_current_density = np.clip(
            unclipped_current_density, 1.0e-3, DEFAULT_MAX_CURRENT_DENSITY
        )

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
