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
COEFFS_RESHAPED = np.load(pth.join(DATA_FOLDER_PATH, npy_file_name))


class PerformancesIronLosses(om.ExplicitComponent):
    """
    Computation of the iron losses of the PMSM results from the magnetic flux density
    constant-variation in magnetic core, obtained from the least square surrogate model (II-67) in
    :cite:`touhami:2020`.
    """

    def initialize(self):
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
            "electrical_frequency",
            units="s**-1",
            val=np.nan,
            shape=number_of_points,
            desc="Number of the north and south pairs in the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
            desc="The magnetic flux density provided by the permanent magnets",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":iron_power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
            desc="Iron loss of the PMSM due to altering magnetic flux",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":iron_power_losses",
            wrt=["electrical_frequency"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":iron_power_losses",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density",
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        w_motor = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass"]
        bm = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density"]
        f = inputs["electrical_frequency"]
        sqrt_f = np.sqrt(f)
        sqrt_bm = np.sqrt(bm)

        # specific power
        sp_pow = 0.0

        for i in range(4):  # i = power of sqrt(f)
            for j in range(4):  # j = power of sqrt(bm)
                coeff = COEFFS_RESHAPED[i][j]
                sp_pow += coeff * (sqrt_f ** (i + 1.0)) * (sqrt_bm ** (j + 1.0))

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":iron_power_losses"] = (
            w_motor * sp_pow / 1000.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        w_motor = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass"]
        bm = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density"]
        f = inputs["electrical_frequency"]
        sqrt_f = np.sqrt(f)
        sqrt_bm = np.sqrt(bm)

        sp_pow = 0.0
        dsp_pow_df = 0.0
        dsp_pow_dbm = 0.0

        for i in range(4):  # i = power of sqrt(f)
            for j in range(4):  # j = power of sqrt(bm)
                coeff = COEFFS_RESHAPED[i][j]
                sp_pow += coeff * (sqrt_f ** (i + 1)) * (sqrt_bm ** (j + 1))
                dsp_pow_df += (
                    coeff * (i + 1.0) * 0.5 * (f ** (0.5 * i - 0.5)) * (sqrt_bm ** (j + 1.0))
                )
                dsp_pow_dbm += (
                    coeff * (sqrt_f ** (i + 1.0)) * (j + 1.0) * 0.5 * (bm ** (0.5 * j - 0.5))
                )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":iron_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass",
        ] = sp_pow / 1000.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":iron_power_losses",
            "electrical_frequency",
        ] = dsp_pow_df * w_motor / 1000.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":iron_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density",
        ] = dsp_pow_dbm * w_motor / 1000.0
