# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesApparentPower(om.ExplicitComponent):
    """Computation of the electric apparent power required to run the motor."""

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

        self.add_input("active_power", units="W", val=np.nan, shape=number_of_points)
        self.add_input(
            "settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":power_factor", val=1.0
        )

        self.add_output(
            "apparent_power",
            units="W",
            val=np.full(number_of_points, 50.0e3),
            shape=number_of_points,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="active_power",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":power_factor",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["apparent_power"] = (
            inputs["active_power"]
            / inputs["settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":power_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        partials["apparent_power", "active_power"] = np.full(
            number_of_points,
            inputs["settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":power_factor"]
            ** -1.0,
        )

        partials[
            "apparent_power",
            "settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":power_factor",
        ] = -(
            inputs["active_power"]
            / inputs["settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":power_factor"]
            ** 2.0
        )
