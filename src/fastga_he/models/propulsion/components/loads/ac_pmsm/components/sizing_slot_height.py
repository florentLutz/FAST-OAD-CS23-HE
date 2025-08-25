# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotHeight(om.ExplicitComponent):
    """
    Computation of single slot height of the PMSM in radial direction. The formula is obtained from
    equation (II-46) in :cite:`touhami:2020.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":current_density_ac_max",
            val=np.nan,
            units="A/m**2",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor",
            val=np.nan,
            desc="The area factor considers the cross-section shape twist due to wire bunching",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor",
            val=np.nan,
            desc="The factor describes the conductor material fullness inside the stator slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress",
            val=np.nan,
            units="N/m**2",
            desc="The tangential tensile strength of the material",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
            desc="The magnetic flux density provided by the permanent magnets",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
            val=np.nan,
            desc="The fraction between overall tooth length and stator bore circumference",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            units="m",
            desc="Single stator slot height (radial)",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        sigma = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress"]
        k_w = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor"]
        b_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        j_rms = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":current_density_ac_max"
        ]
        k_sc = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor"
        ]
        k_fill = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor"]
        r_tooth = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio"]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"] = (
            np.sqrt(2.0) * sigma / (k_w * b_m * j_rms * k_sc * k_fill * (1.0 - r_tooth))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        sigma = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress"]
        k_w = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor"]
        b_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        j_rms = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":current_density_ac_max"
        ]
        k_sc = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor"
        ]
        k_fill = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor"]
        r_tooth = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress",
        ] = np.sqrt(2.0) / (k_w * b_m * j_rms * k_sc * k_fill * (1.0 - r_tooth))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor",
        ] = -np.sqrt(2.0) * sigma / (k_w**2.0 * b_m * j_rms * k_sc * k_fill * (1.0 - r_tooth))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
        ] = -np.sqrt(2.0) * sigma / (k_w * b_m**2.0 * j_rms * k_sc * k_fill * (1.0 - r_tooth))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":current_density_ac_max",
        ] = -np.sqrt(2.0) * sigma / (k_w * b_m * j_rms**2.0 * k_sc * k_fill * (1.0 - r_tooth))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor",
        ] = -np.sqrt(2.0) * sigma / (k_w * b_m * j_rms * k_sc**2.0 * k_fill * (1.0 - r_tooth))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor",
        ] = -np.sqrt(2.0) * sigma / (k_w * b_m * j_rms * k_sc * k_fill**2.0 * (1.0 - r_tooth))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
        ] = np.sqrt(2.0) * sigma / (k_w * b_m * j_rms * k_sc * k_fill * (1.0 - r_tooth) ** 2.0)
