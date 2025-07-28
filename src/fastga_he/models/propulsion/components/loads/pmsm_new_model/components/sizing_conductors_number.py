# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingConductorsNumber(om.ExplicitComponent):
    """
    Computation of the Conductors number.

    """

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

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

        self.add_output(
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number"
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number",
            wrt=[
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
        m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]

        N_c = 2 * p * q * m


        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number"] = N_c

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
    #     m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
    #     p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
    #
    #
    #     N_c = 2 * p * q * m
    #
    #
    #     dR_dp = 2 * q * m
    #     dR_dq = 2 * p * m
    #     dR_dm = 2 * p * q
    #
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
    #     ] = dR_dp
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
    #     ] = dR_dq
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
    #     ] = dR_dm

