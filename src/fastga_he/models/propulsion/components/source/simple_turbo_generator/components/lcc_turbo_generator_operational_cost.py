# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO


import numpy as np
import openmdao.api as om


class LCCTurboGeneratorOperationalCost(om.ExplicitComponent):
    """
    Cost computation of the turbo generator operational cost is based on its purchase price and the
    expected lifespan from https://naasco.com/starter-generators/.
    """

    def initialize(self):
        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )

    def setup(self):
        turbo_generator_id = self.options["turbo_generator_id"]

        self.add_input(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":cost_per_unit",
            units="USD",
            val=np.nan,
        )

        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":operational_cost",
            units="USD/yr",
            val=5.0e2,
        )
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]
        cost = inputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":cost_per_unit"
        ]

        outputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":operational_cost"
        ] = cost * flight_hour / 1250.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turbo_generator_id = self.options["turbo_generator_id"]
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]
        cost = inputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":cost_per_unit"
        ]

        partials[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":operational_cost",
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":cost_per_unit",
        ] = flight_hour / 1250.0

        partials[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = cost / 1250.0
