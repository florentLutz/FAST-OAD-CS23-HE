# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorWindingWeight(om.ExplicitComponent):
    """Computation of the stator winding weight of the PMSM."""

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        # self.options.declare(
        # "diameter_ref",
        # default=0.268,
        # desc="Diameter of the reference motor in [m]",
        # )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_mat_density",
            val=np.nan,
            units="kg/m**3",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":insul_mat_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            units="kg",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_mat_density",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":insul_mat_density",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        lc = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length"]
        hs = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width"]
        rho_c = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_mat_density"]
        rho_insl = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":insul_mat_density"
        ]
        k_fill = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor"]
        ns = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number"]

        vol_wind = lc * hs * ns * ls
        mat_mix_density = k_fill * rho_c + (1.0 - k_fill) * rho_insl

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight"] = (
            vol_wind * mat_mix_density
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        lc = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length"]
        hs = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width"]
        rho_c = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_mat_density"]
        rho_insl = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":insul_mat_density"
        ]
        k_fill = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor"]

        ns = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number"]

        vol_wind = lc * hs * ns * ls
        mat_mix_density = k_fill * rho_c + (1.0 - k_fill) * rho_insl

        # Equation II-46: Slot height hs

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
        ] = hs * ns * ls * mat_mix_density

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
        ] = lc * ns * ls * mat_mix_density

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
        ] = lc * hs * ns * mat_mix_density

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
        ] = lc * hs * ls * mat_mix_density

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor",
        ] = vol_wind * (rho_c - rho_insl)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_mat_density",
        ] = vol_wind * (k_fill)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_winding_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":insul_mat_density",
        ] = vol_wind * (1.0 - k_fill)
