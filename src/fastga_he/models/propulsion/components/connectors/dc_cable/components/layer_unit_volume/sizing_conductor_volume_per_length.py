# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingConductorVolumePerLength(om.ExplicitComponent):
    """Computation of volume per length of conductor layer."""

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

        self.add_output(
            name="conductor_volume_per_length",
            units="m**3",
            val=2.0,
        )

    def setup_partials(self):
        harness_id = self.options["harness_id"]

        self.declare_partials(
            of="conductor_volume_per_length",
            wrt="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:radius",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        outputs["conductor_volume_per_length"] = np.pi * (
            inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":conductor:radius"
            ]
            ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        partials[
            "conductor_volume_per_length",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = (
            2.0
            * np.pi
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":conductor:radius"
            ]
        )
