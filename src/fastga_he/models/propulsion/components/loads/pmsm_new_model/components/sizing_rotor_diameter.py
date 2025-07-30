# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio", val=np.nan
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        x = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio"]
        D = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"]
        # Equation II-43: Stator inner radius R
        D_r = x * D

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter"] = D_r

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness"] = (
            (1 - x) * D / 2
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio",
        ] = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"]

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
        ] = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio"]

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio",
        ] = (-inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"]) / 2

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
        ] = (1 - inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio"]) / 2
