# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class perfSpecificPower(om.ExplicitComponent):
    """Computation of the specific power."""

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass",
            val=np.nan,
            units="kg",
        )

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
            val=np.nan,
            units="W",
            shape=number_of_points,
        )

        self.add_output(name="specific_power", units="W/kg", shape=number_of_points)

        self.declare_partials(
            of="specific_power",
            wrt="shaft_power_out",
            method="fd",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="specific_power",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
            method="fd",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="specific_power",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass",
            method="fd",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["specific_power"] = (
            inputs["shaft_power_out"]
            - inputs[
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses"
            ]
        ) / inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass"]

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     number_of_points = self.options["number_of_points"]
    #
    #     partials[
    #         "specific_power",
    #         "shaft_power_out",
    #     ] = (1 / inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass"]) * np.ones(
    #         number_of_points
    #     )
    #
    #     partials[
    #         "specific_power",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
    #     ] = (-1 / inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass"]) * np.ones(
    #         number_of_points
    #     )
    #
    #     partials[
    #         "specific_power",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass",
    #     ] = (
    #         -(
    #             inputs["shaft_power_out"]
    #             - inputs[
    #                 "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses"
    #             ]
    #         )
    #         / (inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass"]) ** 2
    #     )
