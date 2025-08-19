# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesFrequency(om.ExplicitComponent):
    """Computation of the frequency from number of pole pairs and rpm."""

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

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )

        self.add_output("frequency", units="s**-1", val=0.0, shape=number_of_points)

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="frequency",
            wrt="rpm",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="frequency",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["frequency"] = (
            inputs["rpm"]
            * inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
            / 60.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        partials[
            "frequency", "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"
        ] = inputs["rpm"] / 60.0

        partials["frequency", "rpm"] = np.full(
            number_of_points,
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
            / 60.0,
        )
