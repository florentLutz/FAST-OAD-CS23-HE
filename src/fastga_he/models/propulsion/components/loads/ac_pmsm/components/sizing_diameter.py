# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorDiameter(om.ExplicitComponent):
    """
    Computation of the stator bore diameter of the PMSM. The formula is obtained from equation (
    II-43) in :cite:`touhami:2020.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":form_coefficient",
            val=np.nan,
            desc="The fraction of stator bore diameter and active length",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress",
            val=np.nan,
            units="N/m**2",
            desc="The tangential tensile strength of the material",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
            val=np.nan,
            units="N*m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            units="m",
            desc="Stator bore diameter of the PMSM",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        lambda_ = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":form_coefficient"]
        sigma = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress"]
        T_max = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating"]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"] = 2.0 * (
            ((lambda_ / (4.0 * np.pi * sigma)) * T_max) ** (1.0 / 3.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        x = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":form_coefficient"]
        y = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress"]
        z = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":form_coefficient",
        ] = (
            2.0
            * (z ** (1.0 / 3.0) * (np.pi * y) ** (2.0 / 3.0))
            / (3.0 * (2.0 ** (2.0 / 3.0)) * np.pi * y * x ** (2.0 / 3.0))
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tangential_stress",
        ] = (
            -2.0
            * (x ** (1.0 / 3.0) * z ** (1.0 / 3.0) * (np.pi * y) ** (2.0 / 3.0))
            / (3.0 * (2.0 ** (2.0 / 3.0)) * np.pi * y**2.0)
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":torque_rating",
        ] = (
            2.0
            * (x ** (1.0 / 3.0) * (np.pi * y) ** (2.0 / 3.0))
            / (3.0 * (2.0 ** (2.0 / 3.0)) * np.pi * y * z ** (2.0 / 3.0))
        )
