# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class CableRadius(om.ExplicitComponent):
    """Computation of mass per length of cable."""

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
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness",
            units="m",
            val=0.2e-2,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
            units="m",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        r_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius"
        ]
        t_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness"
        ]
        t_shield = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness"
        ]
        t_sheath = inputs["settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness"]

        r_cable = r_c + t_in + t_shield + t_sheath

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        ] = r_cable

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        output_str = (
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        )

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = 1.0
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
        ] = 1.0
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
        ] = 1.0
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:sheath:thickness",
        ] = 1.0
