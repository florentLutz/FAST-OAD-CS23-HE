# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

# There is a problem with that model in that if the input shaft power goes to 0 so does the
# efficiency which then cause the apparent power to not be 0, cutting off too small value
# should solve the problem
CUTOFF_ETA_MIN = 0.5
CUTOFF_ETA_MAX = 1.0


class PerformancesEfficiencyNew(om.ExplicitComponent):
    """Computation of the efficiency from shaft power and power losses."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            units="W",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":iron_power_losses",
            units="W",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":mechanical_power_losses",
            units="W",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output(
            "efficiency",
            val=np.full(number_of_points, 0.98),
            shape=number_of_points,
            lower=0.0,
            upper=1.0,
        )

        self.declare_partials(
            of="efficiency",
            wrt=[
                "shaft_power_out",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":iron_power_losses",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":mechanical_power_losses",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]
        Pem = inputs["shaft_power_out"]
        P_j = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses"]
        P_iron = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":iron_power_losses"]
        P_mec_loss = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":mechanical_power_losses"
        ]
        rotor_losses_factor = 2  # to take into account the total electromagnetic rotor losses
        P_losses = rotor_losses_factor * (P_mec_loss + P_iron + P_j)
        #
        Pele = Pem + P_losses
        # Mechanical output power
        Pmec = Pem

        # efficiency = Pmec / Pele

        unclipped_efficiency = np.where(
            Pmec != 0.0,
            Pmec / Pele,
            np.ones_like(Pem),
        )

        outputs["efficiency"] = np.clip(unclipped_efficiency, CUTOFF_ETA_MIN, CUTOFF_ETA_MAX)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        Pem = inputs["shaft_power_out"]
        P_j = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses"]
        P_iron = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":iron_power_losses"]
        P_mec_loss = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":mechanical_power_losses"
        ]
        rotor_losses_factor = 2  # to take into account the total electromagnetic rotor losses
        P_losses = rotor_losses_factor * (P_mec_loss + P_iron + P_j)
        #
        Pele = Pem + P_losses
        # Mechanical output power
        Pmec = Pem

        unclipped_efficiency = np.where(
            Pmec != 0.0,
            Pmec / Pele,
            np.ones_like(Pem),
        )

        partials["efficiency", "shaft_power_out"] = np.where(
            (unclipped_efficiency <= 1.0) & (unclipped_efficiency >= 0.5),
            P_losses / (Pele) ** 2.0,
            np.full_like(inputs["shaft_power_out"], 1e-6),
        )

        partials[
            "efficiency", "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses"
        ] = np.where(
            (unclipped_efficiency <= 1.0) & (unclipped_efficiency >= 0.5),
            -(Pmec / (Pele) ** 2.0),
            np.full_like(inputs["shaft_power_out"], 1e-6),
        )

        partials[
            "efficiency", "data:propulsion:he_power_train:PMSM:" + motor_id + ":iron_power_losses"
        ] = np.where(
            (unclipped_efficiency <= 1.0) & (unclipped_efficiency >= 0.5),
            -(Pmec / (Pele) ** 2.0),
            np.full_like(inputs["shaft_power_out"], 1e-6),
        )

        partials[
            "efficiency",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":mechanical_power_losses",
        ] = np.where(
            (unclipped_efficiency <= 1.0) & (unclipped_efficiency >= 0.5),
            -(Pmec / (Pele) ** 2.0),
            np.full_like(inputs["shaft_power_out"], 1e-6),
        )
