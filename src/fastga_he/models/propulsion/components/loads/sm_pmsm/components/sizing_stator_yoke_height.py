# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorYokeHeight(om.ExplicitComponent):
    """
    Computation of the stator yoke thickness of the SM PMSM. The formula is obtained from
    equation (II-45) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pairs in the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yoke_flux_density_ratio",
            val=1.33,
            desc="Maximum mean yoke magnetic flux density divided by the maximum air gap flux "
            "density",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
            val=np.nan,
            units="T",
            desc="Maximum magnetic flux density provided by the permanent magnets",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max",
            val=np.nan,
            units="T",
            desc="The maximum total magnetic flux density in the motor air gap",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            units="m",
            desc="Stator yoke thickness of the SM PMSM",
            val=0.0213,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        air_gap_flux_density_max = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max"
        ]
        max_total_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max"
        ]
        yoke_flux_density_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yoke_flux_density_ratio"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height"] = (
            bore_diameter
            * max_total_flux_density
            / (2.0 * num_pole_pairs * air_gap_flux_density_max * yoke_flux_density_ratio)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        air_gap_flux_density_max = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max"
        ]
        max_total_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max"
        ]
        yoke_flux_density_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yoke_flux_density_ratio"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = max_total_flux_density / (
            2.0 * num_pole_pairs * air_gap_flux_density_max * yoke_flux_density_ratio
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = (
            -bore_diameter
            * max_total_flux_density
            / (2.0 * num_pole_pairs**2.0 * air_gap_flux_density_max * yoke_flux_density_ratio)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":total_flux_density_max",
        ] = bore_diameter / (
            2.0 * num_pole_pairs * air_gap_flux_density_max * yoke_flux_density_ratio
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yoke_flux_density_ratio",
        ] = (
            -bore_diameter
            * max_total_flux_density
            / (2.0 * num_pole_pairs * air_gap_flux_density_max * yoke_flux_density_ratio**2.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
        ] = (
            -bore_diameter
            * max_total_flux_density
            / (2.0 * num_pole_pairs * air_gap_flux_density_max**2.0 * yoke_flux_density_ratio)
        )
