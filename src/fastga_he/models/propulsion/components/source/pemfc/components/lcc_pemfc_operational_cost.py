# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPEMFCStackOperationalCost(om.ExplicitComponent):
    """
    Computation of the PEMFC stack annual operational cost based on the PEMFC lifespan from
    :cite:`fuhren:2022`.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":purchase_cost",
            units="USD",
            val=np.nan,
            desc="Purchase cost of PEMFC stack",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":lifespan_flight_hours",
            units="h",
            val=12.5e3,
            desc="The PEMFC stack lifespan in hours",
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=np.nan,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operational_cost",
            units="USD/yr",
            val=1.0e3,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operational_cost"
        ] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":purchase_cost"
            ]
            * inputs["data:TLAR:flight_hours_per_year"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":lifespan_flight_hours"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operational_cost",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":purchase_cost",
        ] = (
            inputs["data:TLAR:flight_hours_per_year"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":lifespan_flight_hours"
            ]
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operational_cost",
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":lifespan_flight_hours",
        ] = (
            -inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":purchase_cost"
            ]
            * inputs["data:TLAR:flight_hours_per_year"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":lifespan_flight_hours"
            ]
            ** 2.0
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":purchase_cost"
            ]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":lifespan_flight_hours"
            ]
        )
