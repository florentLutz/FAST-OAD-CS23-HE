# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSheathVolumePerLength(om.ExplicitComponent):
    """Computation of volume per length of sheath layer."""

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
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
            units="m",
            val=0.2e-2,
        )

        self.add_output(
            name="sheath_volume_per_length",
            units="m**3",
            val=2.0,
        )

    def setup_partials(self):
        self.declare_partials(of="sheath_volume_per_length", wrt="*", method="exact")

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
        t_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness"
        ]

        outputs["sheath_volume_per_length"] = np.pi * (
            (2.0 * (r_c + t_in + t_shield) + t_sheath) * t_sheath
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
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
        t_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness"
        ]

        partials[
            "sheath_volume_per_length",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = 2.0 * np.pi * t_sheath
        partials[
            "sheath_volume_per_length",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
        ] = 2.0 * np.pi * t_sheath
        partials[
            "sheath_volume_per_length",
            "settings:propulsion:he_power_train:DC_cable_harness:shielding_tape:thickness",
        ] = 2.0 * np.pi * t_sheath
        partials[
            "sheath_volume_per_length",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:thickness",
        ] = 2.0 * np.pi * (r_c + t_in + t_shield + t_sheath)
