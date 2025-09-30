# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotHeight(om.ExplicitComponent):
    """
    Computation of single slot height of the SM PMSM in radial direction. The formula is obtained
    from equation (II-46) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":current_density_ac_caliber",
            val=np.nan,
            units="A/m**2",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor",
            val=np.nan,
            desc="The factor considers the cable winding effect in the stator slot",
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
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":tangential_stress_caliber",
            val=np.nan,
            units="Pa",
            desc="The maximum surface tangential stress applied on rotor by electromagnetism",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
            val=np.nan,
            units="T",
            desc="The magnetic flux density provided by the permanent magnets",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            val=np.nan,
            desc="The fraction between overall tooth length and stator bore circumference",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            units="m",
            desc="Single stator slot height (radial)",
            val=0.035,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        sigma = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
        ]
        winding_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor"
        ]
        magnetic_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max"
        ]
        max_current_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_caliber"
        ]
        slot_conductor_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor"
        ]
        slot_fill_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"
        ]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"] = (
            np.sqrt(2.0)
            * sigma
            / (
                winding_factor
                * magnetic_flux_density
                * max_current_density
                * slot_conductor_factor
                * slot_fill_factor
                * (1.0 - tooth_ratio)
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        sigma = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
        ]
        winding_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor"
        ]
        magnetic_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max"
        ]
        max_current_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_caliber"
        ]
        slot_conductor_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor"
        ]
        slot_fill_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"
        ]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber",
        ] = np.sqrt(2.0) / (
            winding_factor
            * magnetic_flux_density
            * max_current_density
            * slot_conductor_factor
            * slot_fill_factor
            * (1.0 - tooth_ratio)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor",
        ] = (
            -np.sqrt(2.0)
            * sigma
            / (
                winding_factor**2.0
                * magnetic_flux_density
                * max_current_density
                * slot_conductor_factor
                * slot_fill_factor
                * (1.0 - tooth_ratio)
            )
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
        ] = (
            -np.sqrt(2.0)
            * sigma
            / (
                winding_factor
                * magnetic_flux_density**2.0
                * max_current_density
                * slot_conductor_factor
                * slot_fill_factor
                * (1.0 - tooth_ratio)
            )
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_caliber",
        ] = (
            -np.sqrt(2.0)
            * sigma
            / (
                winding_factor
                * magnetic_flux_density
                * max_current_density**2.0
                * slot_conductor_factor
                * slot_fill_factor
                * (1.0 - tooth_ratio)
            )
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor",
        ] = (
            -np.sqrt(2.0)
            * sigma
            / (
                winding_factor
                * magnetic_flux_density
                * max_current_density
                * slot_conductor_factor**2.0
                * slot_fill_factor
                * (1.0 - tooth_ratio)
            )
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
        ] = (
            -np.sqrt(2.0)
            * sigma
            / (
                winding_factor
                * magnetic_flux_density
                * max_current_density
                * slot_conductor_factor
                * slot_fill_factor**2.0
                * (1.0 - tooth_ratio)
            )
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
        ] = (
            np.sqrt(2.0)
            * sigma
            / (
                winding_factor
                * magnetic_flux_density
                * max_current_density
                * slot_conductor_factor
                * slot_fill_factor
                * (1.0 - tooth_ratio) ** 2.0
            )
        )
