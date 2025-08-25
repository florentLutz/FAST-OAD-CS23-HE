# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

# There is a problem with that model in that if the input shaft power goes to 0 so does the
# efficiency which then cause the apparent power to not be 0, cutting off too small value
# should solve the problem


class PerformancesPowerLosses(om.ExplicitComponent):
    """ Computation of the total power losses as sum of Mechanical, Iron, Joule losses."""

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Joule_power_losses",
            units="W",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":iron_power_losses",
            units="W",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
            units="W",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output(
            "power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
        )

        self.declare_partials(
            of="power_losses",
            wrt=[
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Joule_power_losses",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":iron_power_losses",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        P_j = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Joule_power_losses"]
        P_iron = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":iron_power_losses"]
        P_mec_loss = inputs[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses"
        ]
        rotor_losses_factor = 2  # to take into account the total electromagnetic rotor losses

        P_losses = rotor_losses_factor * (P_mec_loss + P_iron + P_j)

        outputs["power_losses"] = P_losses / 1000.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pmsm_id = self.options["pmsm_id"]

        rotor_losses_factor = 2  # to take into account the total electromagnetic rotor losses

        partials[
            "power_losses",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Joule_power_losses",
        ] = rotor_losses_factor * np.ones(number_of_points) / 1000.0

        partials[
            "power_losses",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":iron_power_losses",
        ] = rotor_losses_factor * np.ones(number_of_points) / 1000.0

        partials[
            "power_losses",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
        ] = rotor_losses_factor * np.ones(number_of_points) / 1000.0
