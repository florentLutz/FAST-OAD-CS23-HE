# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHarnessContactorMass(om.ExplicitComponent):
    """
    Computation of the weight of the contactors, based on a regression performed on the data from
    :cite:`giraud:2014`. Assumes that there are 2 contactors per cable. Correlation can be
    found in ...inverter.methodology.contactor_mass.py.
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
            + ":cable:current_caliber",
            units="A",
            val=np.nan,
            desc="Maximum current flowing in one cable of the harness",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":contactor:mass",
            units="kg",
            val=4.5,
            desc="Mass of all the contactors in the harness",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:current_caliber"
        ]
        nb_of_cables = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
        ]

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":contactor:mass"
        ] = 2.0 * nb_of_cables * 0.0239 * current_caliber**0.6942

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:current_caliber"
        ]
        nb_of_cables = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
        ]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":contactor:mass",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:current_caliber",
        ] = 2.0 * nb_of_cables * 0.0239 * 0.6942 * current_caliber ** (0.6942 - 1.0)
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":contactor:mass",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = 2.0 * 0.0239 * current_caliber**0.6942
