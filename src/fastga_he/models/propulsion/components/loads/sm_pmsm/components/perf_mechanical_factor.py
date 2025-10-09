# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import logging

_LOGGER = logging.getLogger(__name__)


class PerformancesMechanicalFactor(om.ExplicitComponent):
    """
    Computation of preventing the output RPM exceeds the centrifugal force limit RPM and the
    first bending mode resonance RPM.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Max continuous rpm of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":poissons_ratio",
            val=0.29,
            desc="The rotor material Poisson's ratio",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yield_stress",
            units="Pa",
            desc="The yield stress of the rotor yoke material, Vacoflux 48 alloy from Vacuumschmelze"
            " materials is set to default",
            val=0.19e9,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":magnetic_material_density",
            val=8.12e3,
            units="kg/m**3",
            desc="The density of soft magnetic material. Vacoflux 48 alloy from Vacuumschmelze "
            "materials is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor",
            val=1.5,
            desc="Safety factor for rotor sizing",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor",
            desc="Remain at 1.0 if the maximum mechanical stress is within the range",
            val=1.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        rho_mag = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        poissons = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":poissons_ratio"]

        rpm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max"]
        sf = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor"]
        sigma = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yield_stress"]

        rpm_mec = 60.0 * np.sqrt(
            8.0 * sigma / (rho_mag * np.pi**2.0 * rotor_diameter**2.0 * (3.0 + poissons))
        )

        if (rpm * sf) > rpm_mec:
            _LOGGER.info(
                msg="Maximum mechanical RPM exceeded. Reduce rotor diameter to stay within "
                "acceptable range."
            )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor"] = (
            np.where((rpm * sf) > rpm_mec, rpm_mec / (rpm * sf), 1.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        rho_mag = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        poissons = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":poissons_ratio"]
        rpm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max"]
        sf = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor"]
        sigma = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yield_stress"]

        rpm_mec = 60.0 * np.sqrt(
            8.0 * sigma / (rho_mag * np.pi**2.0 * rotor_diameter**2.0 * (3.0 + poissons))
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":yield_stress",
        ] = np.where(
            rpm > rpm_mec,
            30.0
            * np.sqrt(8.0 / (rho_mag * np.pi**2.0 * rotor_diameter**2.0 * (3.0 + poissons) * sigma))
            / (sf * rpm),
            0.0,
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor",
        ] = np.where((rpm * sf) > rpm_mec, -rpm_mec / (rpm * sf**2.0), 0.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":poissons_ratio",
        ] = np.where(
            (rpm * sf) > rpm_mec,
            -30.0
            * np.sqrt(
                8.0 * sigma / (rho_mag * np.pi**2.0 * rotor_diameter**2.0 * (3.0 + poissons) ** 3.0)
            )
            / (rpm * sf),
            0.0,
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":magnetic_material_density",
        ] = np.where(
            (rpm * sf) > rpm_mec,
            -30.0
            * np.sqrt(
                8.0 * sigma / (rho_mag**3.0 * np.pi**2.0 * rotor_diameter**2.0 * (3.0 + poissons))
            )
            / (rpm * sf),
            0.0,
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = np.where(
            (rpm * sf) > rpm_mec,
            -60.0
            * np.sqrt(8.0 * sigma / (rho_mag * np.pi**2.0 * rotor_diameter**4.0 * (3.0 + poissons)))
            / (rpm * sf),
            0.0,
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
        ] = np.where((rpm * sf) > rpm_mec, -rpm_mec / (rpm**2.0 * sf), 0.0)
