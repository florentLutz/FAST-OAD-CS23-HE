# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesResistance(om.ExplicitComponent):
    """
    Computation of the electrical resistance (all phases). The formula is obtained from equation (
    II-64) in :cite:`touhami:2020`.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            val=np.nan,
            units="m**-1",
            desc="Total length of the conductor wire divided by the wire cross-sectional area",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistivity",
            shape=number_of_points,
            val=np.nan,
            units="ohm*m",
            desc="Copper electrical resistivity",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance",
            units="ohm",
            shape=number_of_points,
            val=1.5,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistivity",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistivity"]
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistivity",
        ] = np.full(
            number_of_points,
            inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor"
            ],
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistivity"]
