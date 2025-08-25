# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

# There is a problem with that model in that if the input shaft power goes to 0 so does the
# efficiency which then cause the apparent power to not be 0, cutting off too small value
# should solve the problem
CUTOFF_ETA_MIN = 0.5
CUTOFF_ETA_MAX = 1.0


class PerformancesEfficiency(om.ExplicitComponent):
    """Computation of the efficiency from shaft power and power losses."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pmsm_id = self.options["pmsm_id"]

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input("power_losses", units="W", val=np.nan, shape=number_of_points)
        self.add_input(
            "settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":k_efficiency",
            val=1.0,
            desc="K factor for the PMSM efficiency",
        )

        self.add_output(
            "efficiency",
            val=np.full(number_of_points, 0.95),
            shape=number_of_points,
            lower=0.0,
            upper=1.0,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt=["shaft_power_out", "power_losses"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":k_efficiency",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        unclipped_efficiency = np.where(
            inputs["shaft_power_out"] != 0.0,
            inputs["settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":k_efficiency"]
            * inputs["shaft_power_out"]
            / (inputs["shaft_power_out"] + inputs["power_losses"]),
            np.ones_like(inputs["shaft_power_out"]),
        )

        outputs["efficiency"] = np.clip(unclipped_efficiency, CUTOFF_ETA_MIN, CUTOFF_ETA_MAX)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        unclipped_efficiency = np.where(
            inputs["shaft_power_out"] != 0.0,
            inputs["settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":k_efficiency"]
            * inputs["shaft_power_out"]
            / (inputs["shaft_power_out"] + inputs["power_losses"]),
            np.ones_like(inputs["shaft_power_out"]),
        )

        partials["efficiency", "shaft_power_out"] = np.where(
            (unclipped_efficiency <= 1.0) & (unclipped_efficiency >= 0.5),
            inputs["settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":k_efficiency"]
            * inputs["power_losses"]
            / (inputs["shaft_power_out"] + inputs["power_losses"]) ** 2.0,
            np.full_like(inputs["shaft_power_out"], 1e-6),
        )

        partials["efficiency", "power_losses"] = np.where(
            (unclipped_efficiency <= 1.0) & (unclipped_efficiency >= 0.5),
            -(
                inputs["settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":k_efficiency"]
                * inputs["shaft_power_out"]
                / (inputs["shaft_power_out"] + inputs["power_losses"]) ** 2.0
            ),
            np.full_like(inputs["shaft_power_out"], 1e-6),
        )

        partials[
            "efficiency", "settings:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":k_efficiency"
        ] = np.where(
            (unclipped_efficiency <= 1.0) & (unclipped_efficiency >= 0.5),
            inputs["shaft_power_out"] / (inputs["shaft_power_out"] + inputs["power_losses"]),
            np.full_like(inputs["shaft_power_out"], 1e-6),
        )
