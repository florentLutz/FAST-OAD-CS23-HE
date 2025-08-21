# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRotorWeight(om.ExplicitComponent):
    """Computation of the rotor weight of the PMSM."""

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            units="kg",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        lm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        r_r = (inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter"]) / 2.0

        if p <= 10.0:
            rho_rot = -431.67 * p + 7932.0
        elif 10.0 < p <= 50.0:
            rho_rot = 1.09 * p**2.0 - 117.45 * p + 4681.0
        else:
            rho_rot = 1600.0

        # Rotor weight
        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight"] = (
            np.pi * r_r**2.0 * lm * rho_rot
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        lm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        r_r = (inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter"]) / 2.0

        if p <= 10.0:
            rho_rot = -431.67 * p + 7932.0
            drho_dp = -431.67
        elif 10.0 < p <= 50.0:
            rho_rot = 1.09 * p**2.0 - 117.45 * p + 4681.0
            drho_dp = 2.0 * 1.09 * p - 117.45
        else:
            rho_rot = 1600.0
            drho_dp = 0.0

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
        ] = np.pi * r_r**2.0 * rho_rot

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
        ] = np.pi * r_r * lm * rho_rot

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
        ] = np.pi * r_r**2.0 * lm * drho_dp
