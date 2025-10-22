# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorWindingWeight(om.ExplicitComponent):
    """
    Computation of the stator winding weight of the SM PMSM. The formula is obtained from
    equation (II-55) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
            val=np.nan,
            desc="Number of conductor slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
            val=0.5,
            desc="The factor describes the conductor material fullness inside the stator slots, "
            "average value across several conductor shapes is set as default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length",
            val=np.nan,
            units="m",
            desc="Single Conductor cable length in one slot",
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
            + ":conductor_material_density",
            val=8960.0,
            units="kg/m**3",
            desc="Electrical conductor material density, copper is set as default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":insulation_material_density",
            val=1420.0,
            units="kg/m**3",
            desc="Electrical insulation material density, kapton is set to default",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            units="kg",
            desc="The winding cable weight in the stator slots of PMSM",
            val=10.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        conductor_cable_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length"
        ]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        slot_width = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width"]
        rho_conduct = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_material_density"
        ]
        rho_insulate = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":insulation_material_density"
        ]
        slot_fill_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"
        ]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        winding_volume = conductor_cable_length * slot_height * num_conductor_slot * slot_width
        material_mix_density = (
            slot_fill_factor * rho_conduct + (1.0 - slot_fill_factor) * rho_insulate
        )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass"] = (
            winding_volume * material_mix_density
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        conductor_cable_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length"
        ]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        slot_width = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width"]
        rho_conduct = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_material_density"
        ]
        rho_insulate = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":insulation_material_density"
        ]
        slot_fill_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"
        ]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        winding_volume = conductor_cable_length * slot_height * num_conductor_slot * slot_width
        material_mix_density = (
            slot_fill_factor * rho_conduct + (1.0 - slot_fill_factor) * rho_insulate
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length",
        ] = slot_height * num_conductor_slot * slot_width * material_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
        ] = conductor_cable_length * num_conductor_slot * slot_width * material_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
        ] = conductor_cable_length * slot_height * num_conductor_slot * material_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
        ] = conductor_cable_length * slot_height * slot_width * material_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
        ] = winding_volume * (rho_conduct - rho_insulate)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_material_density",
        ] = winding_volume * slot_fill_factor

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":insulation_material_density",
        ] = winding_volume * (1.0 - slot_fill_factor)
