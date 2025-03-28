# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesSFC(om.ExplicitComponent):
    """
    Computation of the ICE sfc for the required MEP. For more information on the method check
    in ...ice_rotax.methodology.fuel_consumption_regression.py
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:max_mep",
            units="g/kW/h",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient",
            val=np.nan,
        )
        self.add_input("mean_effective_pressure", units="bar", val=np.nan, shape=number_of_points)

        self.add_output(
            "specific_fuel_consumption", units="g/kW/h", val=300.0, shape=number_of_points
        )

        self.declare_partials(
            of="specific_fuel_consumption",
            wrt=["mean_effective_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="specific_fuel_consumption",
            wrt="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="specific_fuel_consumption",
            wrt="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:max_mep",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        mep = inputs["mean_effective_pressure"]

        # Theoretically the formula has been developed assuming the MEP is clipped at 18.0 but
        # given the trend, it doesn't cost us anything to not clip it
        clipped_mep = np.clip(mep, 5.0, None)

        sfc_max_mep = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:max_mep"
        ]
        k_coefficient = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient"
        ]

        sfc = sfc_max_mep + np.exp(-k_coefficient * (clipped_mep - 18.0))

        outputs["specific_fuel_consumption"] = sfc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        mep = inputs["mean_effective_pressure"]
        clipped_mep = np.clip(mep, 5.0, None)
        k_coefficient = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient"
        ]

        partials[
            "specific_fuel_consumption",
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":sfc_coefficient:k_coefficient",
        ] = -(clipped_mep - 18.0) * np.exp(-k_coefficient * (clipped_mep - 18.0))
        partials["specific_fuel_consumption", "mean_effective_pressure"] = np.where(
            mep == clipped_mep, -k_coefficient * np.exp(-k_coefficient * (clipped_mep - 18.0)), 1e-6
        )
