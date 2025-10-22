# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorCoreWeight(om.ExplicitComponent):
    """
    Computation of the stator core weight of the SM PMSM. The formula is obtained from
    equation (II-54) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
            val=np.nan,
            units="m",
            desc="The outer stator diameter of the SM PMSM",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area",
            units="m**2",
            val=np.nan,
            desc="Single stator slot section area",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":magnetic_material_density",
            val=8120.0,
            units="kg/m**3",
            desc="The density of soft magnetic material, VACOFLUX 48 from Vacuumschmelze is set "
            "as default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
            desc="Number of conductor slots on the motor stator",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            units="kg",
            val=25.0,
            desc="The weight of the stator excluding the wire weight",
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
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        slot_section_area = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id +
                         ":slot_section_area"]
        rho_magnet = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density"
        ]
        num_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass"] = (
            np.pi * active_length * (stator_radius**2.0 - bore_radius**2.0)
            - (slot_section_area * active_length * num_slot)
        ) * rho_magnet

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        bore_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] / 2.0
        )
        stator_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter"] / 2.0
        )
        num_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        slot_section_area = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id +
                                   ":slot_section_area"]
        rho_magnet = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density"
        ]

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
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area",
        ] = -active_length * num_slot * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = (
            np.pi * (stator_radius**2.0 - bore_radius**2.0) - slot_section_area * num_slot
        ) * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
        ] = -slot_section_area * active_length * rho_magnet

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density",
        ] = np.pi * active_length * (stator_radius**2.0 - bore_radius**2.0) - (
            slot_section_area * active_length * num_slot
        )
