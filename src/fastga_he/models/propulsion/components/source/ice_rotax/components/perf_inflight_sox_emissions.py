# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesHighRPMICEInFlightSOxEmissions(om.ExplicitComponent):
    """
    Computation of the ICE in flight SOx emissions, will be based on a simple emissions index whose
    default value is taken from :cite:`kalivoda:1997`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("fuel_consumed_t", np.full(number_of_points, np.nan), units="kg")
        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":emission_index:SOx",
            units="g/kg",
            val=0.42,
        )

        self.add_output("SOx_emissions", np.full(number_of_points, 0.42), units="g")

        self.declare_partials(
            of="SOx_emissions",
            wrt="fuel_consumed_t",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="SOx_emissions",
            wrt="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":emission_index:SOx",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        outputs["SOx_emissions"] = (
            inputs["fuel_consumed_t"]
            * inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":emission_index:SOx"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        partials["SOx_emissions", "fuel_consumed_t"] = np.full_like(
            inputs["fuel_consumed_t"],
            inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":emission_index:SOx"
            ],
        )
        partials[
            "SOx_emissions",
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":emission_index:SOx",
        ] = inputs["fuel_consumed_t"]
