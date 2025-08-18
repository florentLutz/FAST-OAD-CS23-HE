# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRotorDiameter(om.ExplicitComponent):
    """Computation of the rotor diameter of a cylindrical PMSM."""

    def initialize(self):
        #  Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio", val=np.nan
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
            units="m",
        )
        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness",
            units="m",
        )

    def setup_partials(self):
        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        x = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio"]
        D = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]
        # Equation II-43: Stator inner radius R

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter"] = x * D

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness"] = (
            (1.0 - x) * D / 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio",
        ] = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
        ] = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio",
        ] = (-inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]) / 2.0

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
        ] = (
            1.0 - inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio"]
        ) / 2.0
