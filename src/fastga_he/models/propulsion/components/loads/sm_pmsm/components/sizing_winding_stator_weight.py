# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorWindingWeight(om.ExplicitComponent):
    """
    Computation of the stator winding weight of the PMSM. The formula is obtained from
    equation (II-55) in :cite:`touhami:2020.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number",
            val=np.nan,
            desc="Number of conductor slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_fill_factor",
            val=np.nan,
            desc="The factor describes the conductor material fullness inside the stator slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length",
            val=np.nan,
            units="m",
            desc="Electrical conductor cable length",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_height",
            val=np.nan,
            units="m",
            desc="Single stator slot height (radial)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_width",
            val=np.nan,
            units="m",
            desc="Single stator slot width (along the circumference)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + pmsm_id
            + ":conductor_material_density",
            val=np.nan,
            units="kg/m**3",
            desc="Electrical conductor material density",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + pmsm_id
            + ":insulation_material_density",
            val=np.nan,
            units="kg/m**3",
            desc="Electrical insulation material density",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            units="kg",
            desc="The winding cable weight in the stator slots of PMSM",
            val=10.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        lc = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length"]
        hs = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_width"]
        rho_c = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_material_density"
        ]
        rho_insl = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":insulation_material_density"
        ]
        k_fill = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_fill_factor"]
        ns = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number"]

        vol_wind = lc * hs * ns * ls
        mat_mix_density = k_fill * rho_c + (1.0 - k_fill) * rho_insl

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight"] = (
            vol_wind * mat_mix_density
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        lc = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length"]
        hs = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_width"]
        rho_c = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_material_density"
        ]
        rho_insl = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":insulation_material_density"
        ]
        k_fill = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_fill_factor"]
        ns = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number"]

        vol_wind = lc * hs * ns * ls
        mat_mix_density = k_fill * rho_c + (1.0 - k_fill) * rho_insl

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length",
        ] = hs * ns * ls * mat_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_height",
        ] = lc * ns * ls * mat_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_width",
        ] = lc * hs * ns * mat_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number",
        ] = lc * hs * ls * mat_mix_density

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_fill_factor",
        ] = vol_wind * (rho_c - rho_insl)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_material_density",
        ] = vol_wind * k_fill

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":insulation_material_density",
        ] = vol_wind * (1.0 - k_fill)
