# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class Sizingx2p(om.ExplicitComponent):
    """Computation of the contribute x^2p used for sizing the slot width and yoke height ."""

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        # self.options.declare(
        # "diameter_ref",
        # default=0.268,
        # desc="Diameter of the reference motor in [m]",
        # )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio", val=np.nan
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":x2p",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":x2p",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":x2p",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio",
            method="fd",
        )


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        x = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        x_2p = x ** (2 * p)

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":x2p"] = x_2p

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #
    #     x = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio"]
    #     p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
    #     x_2p = x ** (2 * p)
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":x2p",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio",
    #     ] = (2 * p) * (x ** (2*p-1))
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":x2p",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
    #     ] = 2 * x_2p * np.log(x)
