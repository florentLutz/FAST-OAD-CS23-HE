# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesStatorYokeFluxDensity(om.ExplicitComponent):
    """
    Computation of the mean stator yoke magnetic flux density of the motor. The formula is obtained
    from equation (II-28) in :cite:`touhami:2020`.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            units="m",
            desc="Stator yoke thickness of the SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            units="m",
            desc="Stator bore diameter of the SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pairs in the SM PMSM",
        )

        self.add_output(
            name="yoke_flux_density",
            val=1.0,
            units="T",
            shape=number_of_points,
            desc="Mean magnetic flux density in the stator yoke layer",
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
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            ],
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        total_flux_density = inputs["total_flux_density"]
        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        yoke_height = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height"
        ]

        outputs["yoke_flux_density"] = (
            bore_diameter * total_flux_density / (2.0 * num_pole_pairs * yoke_height)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        total_flux_density = inputs["total_flux_density"]
        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        yoke_height = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height"
        ]

        partials["yoke_flux_density", "total_flux_density"] = np.full(
            number_of_points, bore_diameter / (2.0 * num_pole_pairs * yoke_height)
        )

        partials[
            "yoke_flux_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = total_flux_density / (2.0 * num_pole_pairs * yoke_height)

        partials[
            "yoke_flux_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = -bore_diameter * total_flux_density / (2.0 * num_pole_pairs**2.0 * yoke_height)

        partials[
            "yoke_flux_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
        ] = -bore_diameter * total_flux_density / (2.0 * num_pole_pairs * yoke_height**2.0)
