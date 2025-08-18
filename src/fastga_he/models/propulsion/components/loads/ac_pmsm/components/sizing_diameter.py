# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorDiameter(om.ExplicitComponent):
    """Computation of the diameter of the stator bore of the PMSM."""

    def initialize(self):
        #  Reference motor : HASTECS project, Sarah Touhami

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress",
            val=np.nan,
            units="N/m**2",
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            val=np.nan,
            units="N*m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        lambda_ = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient"]
        sigma = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress"]
        T_max = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating"]

        # Equation II-43: Stator inner radius R
        D = 2 * (((lambda_ / (4 * np.pi * sigma)) * T_max) ** (1 / 3))

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"] = D

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        x = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient"]
        y = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress"]
        z = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient",
        ] = (
            2
            * (z ** (1 / 3) * (np.pi * y) ** (2 / 3))
            / (3 * (2 ** (2 / 3)) * np.pi * y * x ** (2 / 3))
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress",
        ] = (
            -2
            * (x ** (1 / 3) * z ** (1 / 3) * (np.pi * y) ** (2 / 3))
            / (3 * (2 ** (2 / 3)) * np.pi * y**2)
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
        ] = (
            2
            * (x ** (1 / 3) * (np.pi * y) ** (2 / 3))
            / (3 * (2 ** (2 / 3)) * np.pi * y * z ** (2 / 3))
        )
