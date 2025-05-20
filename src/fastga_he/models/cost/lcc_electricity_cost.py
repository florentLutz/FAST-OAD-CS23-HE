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
        self.options.declare("electricity_components_type", types=list, default=[])
        self.options.declare("electricity_components_name", types=list, default=[])

    def setup(self):
        electricity_components_type = self.options["electricity_components_type"]
        electricity_components_name = self.options["electricity_components_name"]

        self.add_input(
            "data:cost:operation:electricity_unit_price",
            units="USD/kW/h",
            val=0.655,
            desc="Price per kW.h of electricity",
        )
        for electricity_storage_type, electricity_storage_id in zip(
            electricity_components_type, electricity_components_name
        ):
            self.add_input(
                "data:propulsion:he_power_train:"
                + electricity_storage_type
                + ":"
                + electricity_storage_id
                + ":energy_consumed_mission",
                units="kW*h",
                val=np.nan,
                desc="Energy drawn from the battery for the mission",
            )

        self.add_output(
            name="data:cost:electricity_cost",
            val=0.0,
            units="USD",
            desc="Electric energy cost for single flight mission",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        electricity_components_type = self.options["electricity_components_type"]
        electricity_components_name = self.options["electricity_components_name"]

        for electricity_storage_type, electricity_storage_id in zip(
            electricity_components_type, electricity_components_name
        ):
            outputs["data:cost:electricity_cost"] += (
                inputs["data:cost:operation:electricity_unit_price"]
                * inputs[
                    "data:propulsion:he_power_train:"
                    + electricity_storage_type
                    + ":"
                    + electricity_storage_id
                    + ":energy_consumed_mission"
                ]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        electricity_components_type = self.options["electricity_components_type"]
        electricity_components_name = self.options["electricity_components_name"]

        for electricity_storage_type, electricity_storage_id in zip(
            electricity_components_type, electricity_components_name
        ):
            partials[
                "data:cost:electricity_cost",
                "data:propulsion:he_power_train:"
                + electricity_storage_type
                + ":"
                + electricity_storage_id
                + ":energy_consumed_mission",
            ] = inputs["data:cost:operation:electricity_unit_price"]

            partials[
                "data:cost:electricity_cost", "data:cost:operation:electricity_unit_price"
            ] += inputs[
                "data:propulsion:he_power_train:"
                + electricity_storage_type
                + ":"
                + electricity_storage_id
                + ":energy_consumed_mission"
            ]
