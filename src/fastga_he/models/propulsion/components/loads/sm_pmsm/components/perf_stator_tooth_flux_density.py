# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesStatorToothFluxDensity(om.ExplicitComponent):
    """
    Computation of the mean stator tooth magnetic flux density of the motor. The formula is obtained
    from equation (II-31) in :cite:`touhami:2020`.
    """

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

        self.add_input(
            name="total_flux_density",
            units="T",
            desc="The total magnetic flux density in air gap including electromagnetism",
            shape=number_of_points,
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            val=np.nan,
            desc="The fraction between overall tooth length and stator bore circumference",
        )

        self.add_output(
            name="tooth_flux_density",
            val=1.3,
            units="T",
            shape=number_of_points,
            desc="Mean magnetic flux density in the stator teeth",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="*",
            wrt="total_flux_density",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            method="exact",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["tooth_flux_density"] = (
            2.0
            * inputs["total_flux_density"]
            / (
                np.pi
                * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        partials["tooth_flux_density", "total_flux_density"] = np.full(
            number_of_points,
            2.0
            / (
                np.pi
                * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]
            ),
        )

        partials[
            "tooth_flux_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
        ] = (
            -2.0
            * inputs["total_flux_density"]
            / (
                np.pi
                * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]
                ** 2.0
            )
        )
