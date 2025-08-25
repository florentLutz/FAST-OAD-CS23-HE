# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingResistance(om.ExplicitComponent):
    """
    Computation of the electrical resistance (all phases). The formula is obtained from equation (
    II-64) in :cite:`touhami:2020.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number",
            val=np.nan,
            desc="Number of conductor slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length",
            val=np.nan,
            units="m",
            desc="Electrical conductor cable length",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_section",
            val=np.nan,
            units="m**2",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistivity",
            val=np.nan,
            units="ohm*m",
            desc="Copper electrical resistivity",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
            units="ohm",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        l_c = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length"]
        s_cond = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_section"]
        ns = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number"]
        rho_cu_twin = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistivity"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance"] = (
            ns * rho_cu_twin * l_c / s_cond
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        l_c = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length"]
        s_cond = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_section"]
        ns = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number"]
        rho_cu_twin = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistivity"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductors_number",
        ] = rho_cu_twin * l_c / s_cond

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_length",
        ] = ns * rho_cu_twin / s_cond

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistivity",
        ] = ns * l_c / s_cond

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":conductor_section",
        ] = -(ns * rho_cu_twin * l_c) / (s_cond**2.0)
