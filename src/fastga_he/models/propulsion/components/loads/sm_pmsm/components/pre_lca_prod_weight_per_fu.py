# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAMotorProdWeightPerFU(om.ExplicitComponent):
    """
    Computation of the weight per functional unit considering the replacement necessary
    during the lifespan of the aircraft. For the default value of the average lifespan of the
    motor, the value is taken from :cite:`thonemann:2024` for short term technologies.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
            units="kg",
            val=np.nan,
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the SM PMSM, typically around 15 year",
        )
        self.add_input(
            name="data:TLAR:aircraft_lifespan",
            val=np.nan,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the SM PMSM required for a functional unit",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="*",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
            ],
            method="exact",
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:TLAR:aircraft_lifespan",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * np.ceil(
                inputs["data:TLAR:aircraft_lifespan"]
                / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan"]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass_per_fu",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan"]
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass"] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":lifespan"]
        )
