# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHighRPMICESFCMinMEP(om.ExplicitComponent):
    """
    Computation of the SFC of the engine for the min value of the mean effective pressure we will
    consider (5 bar). This value is used in the model for the engine sfc which is further described
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
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL",
            units="kW",
            val=np.nan,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:min_mep",
            units="g/kW/h",
            val=619.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:min_mep",
            wrt="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":power_rating_SL",
            val=21.47,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        max_power = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL"
        ]

        sfc_min_mep = 383.28 + 21.47 * (max_power - 58.0)

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:min_mep"
        ] = sfc_min_mep
