# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import IRON_LOSSES_COEFF


class PerformancesIronLosses(om.ExplicitComponent):
    """
    Computation of the iron losses of the SM PMSM results from constant magnetic flux density
    variation in magnetic core, obtained from the least square surrogate model (II-67) in
    :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "electrical_frequency",
            units="Hz",
            val=np.nan,
            shape=number_of_points,
            desc="The oscillation frequency of the SM PMSM AC current",
        )
        self.add_input(
            name="air_gap_flux_density",
            units="T",
            val=np.nan,
            shape=number_of_points,
            desc="The magnetic flux density provided by the permanent magnets",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "iron_power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
            desc="Iron losses of the SM PMSM due to altering magnetic flux",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="iron_power_losses",
            wrt=["electrical_frequency", "air_gap_flux_density"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="iron_power_losses",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        mass = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass"]
        bm = inputs["air_gap_flux_density"]
        f = inputs["electrical_frequency"]

        # Pre-calculate common terms
        f_powers = np.sqrt(f) ** np.arange(1, 5)[:, np.newaxis]
        bm_powers = np.sqrt(bm) ** np.arange(1, 5)[:, np.newaxis]

        # IRON_LOSSES_COEFF needs to be reshaped for broadcasting
        IRON_LOSSES_COEFF_np = np.array(IRON_LOSSES_COEFF)

        # Broadcasting automatically handles the multiplication
        outputs["iron_power_losses"] = (
            mass
            * np.sum(
                IRON_LOSSES_COEFF_np[:, :, np.newaxis]
                * f_powers[:, np.newaxis, :]
                * bm_powers[np.newaxis, :, :],
                axis=(0, 1),
            )
            / 1000.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        mass = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass"]
        bm = inputs["air_gap_flux_density"]
        f = inputs["electrical_frequency"]
        sqrt_f = np.sqrt(f)
        sqrt_bm = np.sqrt(bm)

        # Create coefficient matrix with extra dimension for broadcasting
        IRON_LOSSES_COEFF_np = np.array(IRON_LOSSES_COEFF)

        bm_powers = sqrt_bm ** np.arange(1, 5)[:, np.newaxis]
        bm_derivs = (np.arange(1, 5) * 0.5)[:, np.newaxis] * (
            bm ** (np.arange(4) * 0.5 - 0.5)[:, np.newaxis]
        )

        f_powers = sqrt_f ** np.arange(1, 5)[:, np.newaxis]
        f_derivs = (np.arange(1, 5) * 0.5)[:, np.newaxis] * (
            f ** (np.arange(4) * 0.5 - 0.5)[:, np.newaxis]
        )
        partials[
            "iron_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
        ] = (
            np.sum(
                IRON_LOSSES_COEFF_np[:, :, np.newaxis]
                * f_powers[:, np.newaxis, :]
                * bm_powers[np.newaxis, :, :],
                axis=(0, 1),
            )
            / 1000.0
        )

        partials[
            "iron_power_losses",
            "electrical_frequency",
        ] = (
            mass
            * np.sum(
                IRON_LOSSES_COEFF_np[:, :, np.newaxis]
                * f_derivs[:, np.newaxis, :]
                * bm_powers[np.newaxis, :, :],
                axis=(0, 1),
            )
            / 1000.0
        )

        partials[
            "iron_power_losses",
            "air_gap_flux_density",
        ] = (
            mass
            * np.sum(
                IRON_LOSSES_COEFF_np[:, :, np.newaxis]
                * f_powers[:, np.newaxis, :]
                * bm_derivs[np.newaxis, :, :],
                axis=(0, 1),
            )
            / 1000.0
        )
