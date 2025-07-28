# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":magn_mat_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":magn_mat_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        R = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"] / 2
        R_out = (
            inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter"] / 2
        )
        q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
        m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
        hs = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width"]
        rho_sf = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":magn_mat_density"]

        # Equation II-46: Slot height hs

        Ns = 2 * p * q * m
        W_stat_core = (np.pi * Lm * (R_out**2 - R**2) - (hs * Lm * Ns * ls)) * rho_sf

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight"] = (
            W_stat_core
        )

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     R = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"] / 2
    #     R_out = (
    #         inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter"] / 2
    #     )
    #     q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
    #     m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
    #     p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
    #     Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
    #     hs = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height"]
    #     ls = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width"]
    #     rho_sf = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":magn_mat_density"]
    #
    #     # Equation II-46: Slot height hs
    #
    #     Ns = 2 * p * q * m
    #     dW_dD = -np.pi * Lm * R * rho_sf
    #     dW_dD_ext = np.pi * Lm * R_out * rho_sf
    #     dW_dhs = -Lm * ls * Ns * rho_sf
    #     dW_dls = -Lm * hs * Ns * rho_sf
    #     dW_dLm = (np.pi * (R_out**2 - R**2) - ls * hs * Ns) * rho_sf
    #     dW_dp = -hs * Lm * ls * 2 * q * m * rho_sf
    #     dW_dq = -hs * Lm * ls * 2 * p * m * rho_sf
    #     dW_dm = -hs * Lm * ls * 2 * p * q * rho_sf
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
    #     ] = dW_dD
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
    #     ] = dW_dD_ext
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
    #     ] = dW_dhs
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
    #     ] = dW_dls
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
    #     ] = dW_dLm
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
    #     ] = dW_dp
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
    #     ] = dW_dq
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
    #     ] = dW_dm
