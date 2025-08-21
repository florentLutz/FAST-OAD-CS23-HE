# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorCoreWeight(om.ExplicitComponent):
    """Computation of the stator core weight of the PMSM."""

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":number_of_phases",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slots_per_poles_phases",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":magn_mat_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            units="kg",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":magn_mat_density",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":number_of_phases",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slots_per_poles_phases",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        r = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"] / 2.0
        r_out = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter"]
            / 2.0
        )
        q = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":number_of_phases"]
        m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slots_per_poles_phases"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        lm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        hs = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width"]
        rho_sf = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":magn_mat_density"]

        # Equation II-46: Slot height hs

        ns = 2.0 * p * q * m
        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight"] = (
            np.pi * lm * (r_out**2.0 - r**2.0) - (hs * lm * ns * ls)
        ) * rho_sf

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        r = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"] / 2
        r_out = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter"] / 2
        )
        q = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":number_of_phases"]
        m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slots_per_poles_phases"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        lm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        hs = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width"]
        rho_sf = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":magn_mat_density"]

        ns = 2.0 * p * q * m

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
        ] = -np.pi * lm * r * rho_sf

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
        ] = np.pi * lm * r_out * rho_sf

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
        ] = -lm * ls * ns * rho_sf

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
        ] = -lm * hs * ns * rho_sf

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
        ] = (np.pi * (r_out**2.0 - r**2.0) - ls * hs * ns) * rho_sf

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
        ] = -hs * lm * ls * 2.0 * q * m * rho_sf

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":number_of_phases",
        ] = -hs * lm * ls * 2.0 * p * m * rho_sf

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_core_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slots_per_poles_phases",
        ] = -hs * lm * ls * 2.0 * p * q * rho_sf
