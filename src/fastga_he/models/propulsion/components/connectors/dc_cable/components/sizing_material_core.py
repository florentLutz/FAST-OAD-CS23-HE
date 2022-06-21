# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

RHO_COPPER = 8.96
RHO_ALU = 2.7

CP_COPPER = 386.0
CP_ALU = 902

ALPHA_COPPER = 0.00393
ALPHA_ALU = 0.004308


class MaterialCore(om.ExplicitComponent):

    """Computation of max current per cable ."""

    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            val=1.0,
            desc="1.0 for copper, 0.0 for aluminium",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:density",
            val=8.96,
            units="g/cm**3",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat",
            val=386.0,
            units="J/degK/kg",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor",
            val=0.00393,
            units="degK**-1",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:density",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        # Linear variation between densities of copper and aluminum to allow for easy partial
        # derivatives computation
        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density"
        ] = (RHO_ALU + (RHO_COPPER - RHO_ALU) * material)

        # Linear variation between densities of copper and aluminum to allow for easy partial
        # derivatives computation
        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat"
        ] = (CP_ALU + (CP_COPPER - CP_ALU) * material)

        # Linear variation between densities of copper and aluminum to allow for easy partial
        # derivatives computation
        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor"
        ] = (ALPHA_ALU + (ALPHA_COPPER - ALPHA_ALU) * material)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = (
            RHO_COPPER - RHO_ALU
        )

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:specific_heat",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = (
            CP_COPPER - CP_ALU
        )

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = (
            ALPHA_COPPER - ALPHA_ALU
        )
