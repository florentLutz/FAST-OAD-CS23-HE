# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorCoreWeight(om.ExplicitComponent):
    """
    Computation of the stator core weight of the PMSM. The formula is obtained from
    equation (II-54) in :cite:`touhami:2020`.
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
            desc="Number of the north and south pairs in the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The stator length of PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
            val=np.nan,
            units="m",
            desc="The outer stator diameter of the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            val=np.nan,
            units="m",
            desc="Single stator slot height (radial)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
            val=np.nan,
            units="m",
            desc="Single stator slot width (along the circumference)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":magnetic_material_density",
            val=np.nan,
            units="kg/m**3",
            desc="The density of soft magnetic material",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            units="kg",
            val=25.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        bore_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] / 2.0
        )
        stator_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter"] / 2.0
        )
        slots_per_poles_phases = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        slot_width = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width"]
        rho_magnet = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density"
        ]
        ns = 6.0 * num_pole_pairs * slots_per_poles_phases

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass"] = (
            np.pi * active_length * (stator_radius**2.0 - bore_radius**2.0)
            - (slot_height * active_length * ns * slot_width)
        ) * rho_magnet

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        bore_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] / 2.0
        )
        stator_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter"] / 2.0
        )
        slots_per_poles_phases = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        slot_width = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width"]
        rho_magnet = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density"
        ]
        ns = 6.0 * num_pole_pairs * slots_per_poles_phases

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = -np.pi * active_length * bore_radius * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
        ] = np.pi * active_length * stator_radius * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
        ] = -active_length * slot_width * ns * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
        ] = -active_length * slot_height * ns * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = (
            np.pi * (stator_radius**2.0 - bore_radius**2.0) - slot_width * slot_height * ns
        ) * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = -slot_height * active_length * slot_width * 6.0 * slots_per_poles_phases * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases",
        ] = -slot_height * active_length * slot_width * 6.0 * num_pole_pairs * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density",
        ] = np.pi * active_length * (stator_radius**2.0 - bore_radius**2.0) - (
            slot_height * active_length * ns * slot_width
        )
