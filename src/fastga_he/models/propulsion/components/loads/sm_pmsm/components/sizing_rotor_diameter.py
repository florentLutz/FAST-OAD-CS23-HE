# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRotorDiameter(om.ExplicitComponent):
    """
    Computation of the rotor diameter and airgap thickness of a cylindrical PMSM. The formulas
    are obtained from equation (II-50) and (II-51) in :cite:`touhami:2020.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":radius_ratio",
            val=np.nan,
            desc="the radius ratio of the rotor radius and the stator bore radius",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rotor_diameter",
            units="m",
        )
        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_thickness",
            units="m",
            desc="The distance between the rotor and the stator bore",
        )

    def setup_partials(self):
        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        x = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":radius_ratio"]
        d = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":diameter"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rotor_diameter"] = x * d
        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_thickness"] = (
            (1.0 - x) * d / 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":radius_ratio",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":diameter"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":diameter",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":radius_ratio"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_thickness",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":radius_ratio",
        ] = (-inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":diameter"]) / 2.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_thickness",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":diameter",
        ] = (
            1.0 - inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":radius_ratio"]
        ) / 2.0
