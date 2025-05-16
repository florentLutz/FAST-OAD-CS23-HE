# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
import logging

import numpy as np
import openmdao.api as om

_LOGGER = logging.getLogger(__name__)


class LCCElectricEnergyCost(om.ExplicitComponent):
    """
    Computation of the electric energy cost of the aircraft for a single mission. The
    charging cost is estimated from https://eniplenitude.eu/e-mobility/pricing.
    """

    def initialize(self):
        self.options.declare("cost_components_type", types=list, default=[])
        self.options.declare("cost_components_name", types=list, default=[])

    def setup(self):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]

        self.add_output(
            name="data:cost:electric_energy_cost",
            val=1000.0,
            units="USD",
            desc="Electric energy cost for single flight mission",
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

            self.declare_partials(of="*", wrt="*", val=0.655)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cost_components_type = self.options["cost_components_type"]
        cost_components_name = self.options["cost_components_name"]
        outputs["data:cost:electric_energy_cost"] = 0.0

        for battery_type, battery_id in [
            (comp_type, comp_name)
            for comp_type, comp_name in zip(cost_components_type, cost_components_name)
            if comp_type == "battery_pack"
        ]:
            outputs["data:cost:electric_energy_cost"] += (
                0.655
                * inputs[
                    "data:propulsion:he_power_train:"
                    + battery_type
                    + ":"
                    + battery_id
                    + ":energy_consumed_mission"
                ]
            )
