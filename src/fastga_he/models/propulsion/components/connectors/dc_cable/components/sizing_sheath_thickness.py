# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCableSheathThickness(om.ExplicitComponent):
    """
    Computation of the thickness of the sheath based on the diameter of the bare cable (without
    sheath). Formula taken from :cite:`stuckl:2016`
    """

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
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:radius",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
            units="m",
            val=0.2e-3,
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
            units="m",
            val=0.2e-2,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        bare_cable_diameter = 2.0 * (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":conductor:radius"
            ]
            + inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":insulation:thickness"
            ]
            + inputs["settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness"]
        )

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness"
        ] = 0.035 * bare_cable_diameter + 1e-3

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = 0.035 * 2.0
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
        ] = 0.035 * 2.0
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
        ] = 0.035 * 2.0
