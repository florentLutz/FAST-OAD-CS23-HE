# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingDCSSPCWeight(om.ExplicitComponent):
    """
    Computation of the weight of the DC SSPC module. For now no detailed breakdown nor
    regression could be obtained so a simple power density will be used. Value will be taken as
    those of a bidirectional SSBC from :cite:`valente:2021`.
    """

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the SSPC",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            units="V",
            val=np.nan,
            desc="Voltage caliber of the SSPC",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":power_density",
            units="W/kg",
            val=30.0e3,
            desc="Power density of the SSPC",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":mass",
            units="kg",
            val=20.0,
            desc="Mass of the SSPC",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"]
            * inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"]
        ) / inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":power_density"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":mass",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
        ] = (
            (inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"])
            / inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":power_density"]
        )
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":mass",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
        ] = (
            (inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"])
            / inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":power_density"]
        )
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":mass",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":power_density",
        ] = (
            -(
                inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"]
                * inputs[
                    "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"
                ]
            )
            / inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":power_density"]
            ** 2.0
        )
