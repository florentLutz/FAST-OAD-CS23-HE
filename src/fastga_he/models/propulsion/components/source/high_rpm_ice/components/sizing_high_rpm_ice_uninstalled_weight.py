# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHighRPMICEUninstalledWeight(om.ExplicitComponent):
    """
    Computation of the uninstalled ICE weight, based on a formula from :cite:`gudmundsson:2013`
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL",
            units="kW",
            val=np.nan,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":uninstalled_mass",
            units="kg",
            val=60.0,
            desc="Uninstalled weight of the ICE engine",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":uninstalled_mass",
            wrt="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":power_rating_SL",
            val=0.109,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        max_power = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL"
        ]
        uninstalled_mass = 0.109 * (max_power - 58.0) + 55.4

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":uninstalled_mass"
        ] = uninstalled_mass
