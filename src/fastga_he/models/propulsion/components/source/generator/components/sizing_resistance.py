# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingGeneratorPhaseResistance(om.ExplicitComponent):
    """Computation of the phase resistance of a cylindrical generator."""

    def initialize(self):
        # Reference generator : EMRAX 268

        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )
        self.options.declare(
            "resistance_ref",
            default=22.9,
            desc="Phase resistance of the reference generator in [mOhm]",
        )

    def setup(self):
        generator_id = self.options["generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":scaling:phase_resistance",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":phase_resistance",
            val=self.options["resistance_ref"],
            units="ohm",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":phase_resistance",
            wrt="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":scaling:phase_resistance",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        resistance_scaling = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:phase_resistance"
        ]

        resistance_ref = self.options["resistance_ref"]

        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":phase_resistance"
        ] = resistance_ref * 1e-3 * resistance_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]

        resistance_ref = self.options["resistance_ref"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":phase_resistance",
            "data:propulsion:he_power_train:generator:"
            + generator_id
            + ":scaling:phase_resistance",
        ] = resistance_ref * 1e-3
