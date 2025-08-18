# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingResistanceNew2(om.ExplicitComponent):
    """
    Computation of the Resistance (all phases).

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_section",
            val=np.nan,
            units="m**2",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistivity",
            val=np.nan,
            units="ohm*m",
        )

        self.add_output(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistance",
            units="ohm",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistance",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_section",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistivity",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        l_c = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length"]
        S_cond = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_section"]
        N_c = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number"]
        rho_cu_Twin = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistivity"]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistance"] = (
            N_c * rho_cu_Twin * l_c / S_cond
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        l_c = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length"]
        S_cond = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_section"]
        N_c = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number"]
        rho_cu_Twin = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistivity"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
        ] = rho_cu_Twin * l_c / S_cond

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
        ] = N_c * rho_cu_Twin / S_cond

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistivity",
        ] = N_c * l_c / S_cond

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_section",
        ] = -(N_c * rho_cu_Twin * l_c) / (S_cond**2.0)
