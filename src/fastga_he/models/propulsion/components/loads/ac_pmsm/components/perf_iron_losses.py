# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import os.path as pth

DATA_FOLDER_PATH = pth.join(
    pth.join(pth.dirname(pth.dirname(__file__)), "methodology"), "iron_losses"
)
npy_file_name = "coeffs_reshaped.npy"
coeffs_reshaped = np.load(pth.join(DATA_FOLDER_PATH, npy_file_name))


class PerformancesIronLosses(om.ExplicitComponent):
    """
    Computation of the Iron losses.

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

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mass",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":iron_power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":iron_power_losses",
            wrt=["rpm"],
            method="fd",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":iron_power_losses",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mass",
            ],
            method="fd",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        RPM = inputs["rpm"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        w_motor = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mass"]
        bm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        f = RPM * p / 60.0
        sqrt_f = np.sqrt(f)
        sqrt_bm = np.sqrt(bm)

        # specific power
        Sp_pow = 0.0

        for i in range(4):  # i = power of sqrt(f)
            for j in range(4):  # j = power of sqrt(bm)
                coeff = coeffs_reshaped[i][j]
                term = coeff * (sqrt_f ** (i + 1)) * (sqrt_bm ** (j + 1))
                Sp_pow += term

        # Specific power times the PMSM weight
        P_iron = w_motor * Sp_pow

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":iron_power_losses"] = (
            P_iron / 1000.0
        )

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     RPM = inputs["rpm"]
    #     p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
    #     w_motor = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mass"]
    #     bm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
    #     f = RPM * p / 60
    #     sqrt_f = np.sqrt(f)
    #     sqrt_bm = np.sqrt(bm)
    #
    #     # partials[
    #     #    "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Joule_power_losses",
    #     #    "ac_current_rms_in_one_phase",
    #     # ] = dP_dIrms
