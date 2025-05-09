# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPropellerCost(om.ExplicitComponent):
    """
    Computation of the propeller purchasing cost from :cite:`gudmundsson:2013`. The propeller
    type selection variable is only applied in cost estimation, does not affect other
    propeller-related calculations.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":shaft_power_in_max",
            units="hp",
            val=np.nan,
            desc="Maximum value of the propeller input shaft power",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":type",
            val=1.0,
            desc="Value set to 1.0 if constant-speed propeller, 0.0 for fixed-pitch propeller. "
            "This is only use in cost estimation, does not affect other propeller-related "
            "models.",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cost_per_unit",
            units="USD",
            val=1000.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]
        cpi_2012 = inputs["data:cost:cpi_2012"]
        shaft_power_max = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":shaft_power_in_max"
        ]
        d_prop = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        prop_type = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":type"]

        fixed_pitch_cost = 3145.0 * cpi_2012
        constant_speed_cost = 209.69 * cpi_2012 * d_prop**2.0 * (shaft_power_max / d_prop) ** 0.12

        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":cost_per_unit"] = (
            1.0 - prop_type
        ) * fixed_pitch_cost + prop_type * constant_speed_cost

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        cpi_2012 = inputs["data:cost:cpi_2012"]
        shaft_power_max = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":shaft_power_in_max"
        ]
        d_prop = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        prop_type = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":type"]

        fixed_pitch_cost = 3145.0 * cpi_2012
        constant_speed_cost = 209.69 * cpi_2012 * d_prop**2.0 * (shaft_power_max / d_prop) ** 0.12

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cost_per_unit",
            "data:cost:cpi_2012",
        ] = (1.0 - prop_type) * 3145.0 + prop_type * 209.69 * d_prop**2.0 * (
            shaft_power_max / d_prop
        ) ** 0.12

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cost_per_unit",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":type",
        ] = constant_speed_cost - fixed_pitch_cost

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cost_per_unit",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":shaft_power_in_max",
        ] = 25.1628 * prop_type * cpi_2012 * d_prop**1.88 / shaft_power_max**0.88

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cost_per_unit",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = 394.2172 * prop_type * cpi_2012 * d_prop**0.88 * shaft_power_max**0.12
