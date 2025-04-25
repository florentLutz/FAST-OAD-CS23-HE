# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPEMFCStackOperation(om.ExplicitComponent):
    """
    Computation of the PEMFC stack operation cost based on the PEMFC lifespan from
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
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":cost_per_unit",
            units="USD",
            val=np.nan,
            desc="Purchase cost of PEMFC stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":lifespan",
            units="h",
            val=12.5e3,
            desc="Purchase cost of PEMFC stack",
        )

        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=np.nan,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operation_cost",
            units="USD/yr",
            val=1.0e3,
            desc="Annual maintenance cost of PEMFC stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operation_cost"
        ] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":cost_per_unit"
            ]
            * inputs["data:TLAR:flight_hours_per_year"]
            / inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        cost = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":cost_per_unit"
        ]
        lifespan = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":lifespan"
        ]
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operation_cost",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":cost_per_unit",
        ] = flight_hour / lifespan

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operation_cost",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":lifespan",
        ] = -cost * flight_hour / lifespan**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":operation_cost",
            "data:TLAR:flight_hours_per_year",
        ] = cost / lifespan
