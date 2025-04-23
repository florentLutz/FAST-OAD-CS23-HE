# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCHighRPMICEOperation(om.ExplicitComponent):
    """
    Computation of ICE engine operation cost from
    http://blog.overhaulbids.com/lycoming-overhaul-cost/.
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
            + ":displacement_volume",
            units="inch**3",
            val=np.nan,
        )

        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":operation_cost",
            units="USD/yr",
            val=1e4,
            desc="Cost of the ICE per unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        volume = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":displacement_volume"
        ]
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":operation_cost"
        ] = (0.103 * volume - 4.41) * flight_hour / 1.8

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        volume = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":displacement_volume"
        ]
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]

        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":operation_cost",
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":displacement_volume",
        ] = 0.103 * flight_hour / 1.8
        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":operation_cost",
            "data:TLAR:flight_hours_per_year",
        ] = (0.103 * volume - 4.41) / 1.8
