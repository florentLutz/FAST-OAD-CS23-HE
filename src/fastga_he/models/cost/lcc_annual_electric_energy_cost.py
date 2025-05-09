# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
import logging

import numpy as np
import openmdao.api as om

_LOGGER = logging.getLogger(__name__)


class LCCAnnualElectricEnergyCost(om.ExplicitComponent):
    """
    Computation of the yearly electric energy cost of the aircraft.
    """

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )

        self.add_output(
            name="data:operation:annual_electric_energy_cost",
            val=1000.0,
            units="USD/yr",
        )

        for battery_type, battery_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if comp_type == "battery_pack"
        ]:
            self.add_input(
                "data:propulsion:he_power_train:"
                + battery_type
                + ":"
                + battery_id
                + ":energy_consumed_mission",
                units="kW*h",
                val=np.nan,
                desc="Energy drawn from the battery for the mission",
            )

            self.declare_partials(
                of="*",
                wrt=[
                    "data:propulsion:he_power_train:"
                    + battery_type
                    + ":"
                    + battery_id
                    + ":energy_consumed_mission",
                    "data:TLAR:flight_per_year",
                ],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]
        outputs["data:operation:annual_electric_energy_cost"] = 0.0

        for battery_type, battery_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if comp_type == "battery_pack"
        ]:
            outputs["data:operation:annual_electric_energy_cost"] += (
                0.655
                * flight_per_year
                * inputs[
                    "data:propulsion:he_power_train:"
                    + battery_type
                    + ":"
                    + battery_id
                    + ":energy_consumed_mission"
                ]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]

        for battery_type, battery_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if comp_type == "battery_pack"
        ]:
            energy_consumed = inputs[
                "data:propulsion:he_power_train:"
                + battery_type
                + ":"
                + battery_id
                + ":energy_consumed_mission"
            ]
            partials["data:operation:annual_electric_energy_cost", "data:TLAR:flight_per_year"] = (
                0.655 * energy_consumed
            )
            partials[
                "data:operation:annual_electric_energy_cost",
                "data:propulsion:he_power_train:"
                + battery_type
                + ":"
                + battery_id
                + ":energy_consumed_mission",
            ] = 0.655 * flight_per_year
