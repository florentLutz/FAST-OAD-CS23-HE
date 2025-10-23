# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesIronLosses(om.ExplicitComponent):
    """
    Computation of the iron losses of the SM PMSM results from design air gap magnetic flux
    density, obtained from the least square surrogate model (II-67) in :cite:`touhami:2020`.
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
            "electrical_frequency",
            units="Hz",
            val=np.nan,
            shape=number_of_points,
            desc="The oscillation frequency of the SM PMSM AC current",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":design_air_gap_flux_density",
            val=np.nan,
            units="T",
            desc="The design air gap magnetic flux density",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "iron_power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
            desc="Iron losses of the SM PMSM due to altering magnetic flux",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="iron_power_losses",
            wrt=["electrical_frequency"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="iron_power_losses",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":design_air_gap_flux_density",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        mass = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass"]
        air_gap_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_air_gap_flux_density"
        ]
        f = inputs["electrical_frequency"]

        outputs["iron_power_losses"] = (
            mass
            * (
                530.850444 * f**0.5 * air_gap_flux_density**0.5
                - 1660.22877 * f**0.5 * air_gap_flux_density
                + 1676.66819 * f**0.5 * air_gap_flux_density**1.5
                - 540.045900 * f**0.5 * air_gap_flux_density**2.0
                - 40.4065802 * f * air_gap_flux_density**0.5
                + 126.706523 * f * air_gap_flux_density
                - 127.987721 * f * air_gap_flux_density**1.5
                + 41.0664456 * f * air_gap_flux_density**2.0
                + 0.843378999 * f**1.5 * air_gap_flux_density**0.5
                - 2.63865343 * f**1.5 * air_gap_flux_density
                + 2.65237021 * f**1.5 * air_gap_flux_density**1.5
                - 0.840175850 * f**1.5 * air_gap_flux_density**2.0
                - 0.00435714286 * f**2.0 * air_gap_flux_density**0.5
                + 0.0135660947 * f**2.0 * air_gap_flux_density
                - 0.0135585345 * f**2.0 * air_gap_flux_density**1.5
                + 0.00425562126 * f**2.0 * air_gap_flux_density**2.0
            )
            / 1000.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        mass = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass"]
        air_gap_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_air_gap_flux_density"
        ]
        f = inputs["electrical_frequency"]

        partials[
            "iron_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
        ] = (
            530.850444 * f**0.5 * air_gap_flux_density**0.5
            - 1660.22877 * f**0.5 * air_gap_flux_density
            + 1676.66819 * f**0.5 * air_gap_flux_density**1.5
            - 540.045900 * f**0.5 * air_gap_flux_density**2.0
            - 40.4065802 * f * air_gap_flux_density**0.5
            + 126.706523 * f * air_gap_flux_density
            - 127.987721 * f * air_gap_flux_density**1.5
            + 41.0664456 * f * air_gap_flux_density**2.0
            + 0.843378999 * f**1.5 * air_gap_flux_density**0.5
            - 2.63865343 * f**1.5 * air_gap_flux_density
            + 2.65237021 * f**1.5 * air_gap_flux_density**1.5
            - 0.840175850 * f**1.5 * air_gap_flux_density**2.0
            - 0.00435714286 * f**2.0 * air_gap_flux_density**0.5
            + 0.0135660947 * f**2.0 * air_gap_flux_density
            - 0.0135585345 * f**2.0 * air_gap_flux_density**1.5
            + 0.00425562126 * f**2.0 * air_gap_flux_density**2.0
        ) / 1000.0

        partials[
            "iron_power_losses",
            "electrical_frequency",
        ] = (
            mass
            * (
                265.425222 * f**-0.5 * air_gap_flux_density**0.5
                - 830.114385 * f**-0.5 * air_gap_flux_density
                + 838.334095 * f**-0.5 * air_gap_flux_density**1.5
                - 270.02295 * f**-0.5 * air_gap_flux_density**2.0
                - 40.4065802 * air_gap_flux_density**0.5
                + 126.706523 * air_gap_flux_density
                - 127.987721 * air_gap_flux_density**1.5
                + 41.0664456 * air_gap_flux_density**2.0
                + 1.2650684985 * f**0.5 * air_gap_flux_density**0.5
                - 3.957980145 * f**0.5 * air_gap_flux_density
                + 3.978555315 * f**0.5 * air_gap_flux_density**1.5
                - 1.260263775 * f**0.5 * air_gap_flux_density**2.0
                - 0.00871428572 * f * air_gap_flux_density**0.5
                + 0.0271321894 * f * air_gap_flux_density
                - 0.027117069 * f * air_gap_flux_density**1.5
                + 0.00851124252 * f * air_gap_flux_density**2.0
            )
            / 1000.0
        )

        partials[
            "iron_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_air_gap_flux_density",
        ] = (
            mass
            * (
                265.425222 * f**0.5 * air_gap_flux_density**-0.5
                - 1660.22877 * f**0.5
                + 2515.002285 * f**0.5 * air_gap_flux_density**0.5
                - 1080.0918 * f**0.5 * air_gap_flux_density
                - 20.2032901 * f * air_gap_flux_density**-0.5
                + 126.706523 * f
                - 191.9815815 * f * air_gap_flux_density**0.5
                + 82.1328912 * f * air_gap_flux_density
                + 0.4216894995 * f**1.5 * air_gap_flux_density**-0.5
                - 2.63865343 * f**1.5
                + 3.978555315 * f**1.5 * air_gap_flux_density**0.5
                - 1.6803517 * f**1.5 * air_gap_flux_density
                - 0.00217857143 * f**2.0 * air_gap_flux_density**-0.5
                + 0.0135660947 * f**2.0
                - 0.02033780175 * f**2.0 * air_gap_flux_density**0.5
                + 0.00851124252 * f**2.0 * air_gap_flux_density
            )
            / 1000.0
        )
