# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHighRPMICEWeight(om.ExplicitComponent):
    """
    Computation of the installed ICE weight, Based on a statistical analysis, see
    :cite:`raymer:2012`.
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
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":uninstalled_mass",
            units="kg",
            val=np.nan,
            desc="Uninstalled weight of the ICE engine",
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":mass",
            units="kg",
            val=70.0,
            desc="Installed weight of the ICE engine",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":mass",
            wrt="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":uninstalled_mass",
            val=1.2,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        # We take 1.2 for the installed mass as now, the propeller weight is properly computed as
        # are the fuel lines
        outputs["data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":mass"] = (
            1.2
            * inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":uninstalled_mass"
            ]
        )
