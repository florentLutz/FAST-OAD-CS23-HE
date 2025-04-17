# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHighRPMICESFCKCoefficient(om.ExplicitComponent):
    """
    The SFC based on the MEP will be computed under the form of a + exp(-k*(MEP-MEP_MAX)) with
    coefficient a nd k computed to match the value of the sfc at the min and max value of the MEP
    computed based on regression on the Rotax 912 family. For more information on the method check
    in ...ice_rotax.methodology.fuel_consumption_regression.py
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
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:min_mep",
            units="g/kW/h",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:max_mep",
            units="g/kW/h",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient",
            val=0.44,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        sfc_min_mep = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:min_mep"
        ]
        sfc_max_mep = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:max_mep"
        ]
        k_coefficient = (
            np.log(sfc_min_mep - sfc_max_mep) / 13.0
        )  # The 13 represents the difference between min MEP (5.0 bar) and max MEP (18.0 bar)

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient"
        ] = k_coefficient

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        sfc_min_mep = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:min_mep"
        ]
        sfc_max_mep = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:max_mep"
        ]

        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient",
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:min_mep",
        ] = 1.0 / (sfc_min_mep - sfc_max_mep) / 13.0
        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient",
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:max_mep",
        ] = -1.0 / (sfc_min_mep - sfc_max_mep) / 13.0
