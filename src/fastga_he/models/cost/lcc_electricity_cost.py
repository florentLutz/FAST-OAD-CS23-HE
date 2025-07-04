# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCElectricityCost(om.ExplicitComponent):
    """
    Computation of the aircraft electricity cost of the aircraft for a single mission. The
    charging cost is estimated from https://eniplenitude.eu/e-mobility/pricing.
    """

    def initialize(self):
        self.options.declare("electricity_components_types", types=list, default=[])
        self.options.declare("electricity_components_names", types=list, default=[])

    def setup(self):
        electricity_components_types = self.options["electricity_components_types"]
        electricity_components_names = self.options["electricity_components_names"]

        self.add_input(
            "data:cost:operation:electricity_unit_price",
            units="USD/kW/h",
            val=0.655,
            desc="Price per kW.h of electricity",
        )
        for electricity_storage_type, electricity_storage_id in zip(
            electricity_components_types, electricity_components_names
        ):
            self.add_input(
                "data:propulsion:he_power_train:"
                + electricity_storage_type
                + ":"
                + electricity_storage_id
                + ":energy_consumed_main_route",
                units="kW*h",
                val=np.nan,
                desc="Energy drawn from the battery for the main_route",
            )

        self.add_output(
            name="data:cost:electricity_cost",
            val=0.0,
            units="USD",
            desc="Electric energy cost for single flight mission",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        electricity_components_types = self.options["electricity_components_types"]
        electricity_components_names = self.options["electricity_components_names"]

        for electricity_storage_type, electricity_storage_id in zip(
            electricity_components_types, electricity_components_names
        ):
            outputs["data:cost:electricity_cost"] += (
                inputs["data:cost:operation:electricity_unit_price"]
                * inputs[
                    "data:propulsion:he_power_train:"
                    + electricity_storage_type
                    + ":"
                    + electricity_storage_id
                    + ":energy_consumed_main_route"
                ]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        electricity_components_types = self.options["electricity_components_types"]
        electricity_components_names = self.options["electricity_components_names"]

        for electricity_storage_type, electricity_storage_id in zip(
            electricity_components_types, electricity_components_names
        ):
            partials[
                "data:cost:electricity_cost",
                "data:propulsion:he_power_train:"
                + electricity_storage_type
                + ":"
                + electricity_storage_id
                + ":energy_consumed_main_route",
            ] = inputs["data:cost:operation:electricity_unit_price"]

            partials[
                "data:cost:electricity_cost", "data:cost:operation:electricity_unit_price"
            ] += inputs[
                "data:propulsion:he_power_train:"
                + electricity_storage_type
                + ":"
                + electricity_storage_id
                + ":energy_consumed_main_route"
            ]
