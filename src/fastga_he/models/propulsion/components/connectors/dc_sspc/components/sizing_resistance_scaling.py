# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingDCSSPCResistanceScaling(om.ExplicitComponent):
    """Computation of scaling ratio for the resistances of the DC SSPC."""

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

        self.options.declare(
            name="current_caliber_ref",
            types=float,
            default=450.0,
            desc="Current caliber of the reference component",
        )

    def setup(self):

        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the SSPC",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":scaling:resistance"
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":scaling:resistance",
            wrt="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_sspc_id = self.options["dc_sspc_id"]

        current_caliber_ref = self.options["current_caliber_ref"]
        current_caliber = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"
        ]

        current_caliber_star = current_caliber / current_caliber_ref

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":scaling:resistance"] = (
            current_caliber_star ** -1
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_sspc_id = self.options["dc_sspc_id"]

        current_caliber_ref = self.options["current_caliber_ref"]
        current_caliber = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"
        ]

        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":scaling:resistance",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
        ] = (
            -current_caliber_ref / current_caliber ** 2.0
        )
