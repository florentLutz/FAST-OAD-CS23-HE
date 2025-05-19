# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging
import numpy as np
import openmdao.api as om

DEFAULT_FUEL_UNIT_COST = {"jet_fuel": 2.967, "diesel": 1.977, "avgas": 3.66, "hydrogen": 6.54}

_LOGGER = logging.getLogger(__name__)


class LCCFuelCost(om.ExplicitComponent):
    """
    Computation of the fuel cost of the aircraft for single mission. The cost of unit hydrogen is
    obtained from :cite:`sens:2024`. The unit price of avgas 100LL and Jet-A1 are obtained from
    https://orleans.aeroport.fr.
    """

    def initialize(self):
        self.options.declare("tank_types", types=list, default=[])
        self.options.declare("tank_names", types=list, default=[])
        self.options.declare("fuel_types", types=list, default=[])

    def setup(self):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            self.add_input(
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_consumed_mission",
                units="kg",
                val=np.nan,
                desc="Amount of fuel from that tank which will be consumed during mission",
            )
            if fuel_type not in DEFAULT_FUEL_UNIT_COST:
                _LOGGER.warning("Fuel type does not exist, replaced by Jet-A1!")
                fuel_type = "jet_fuel"
            self.add_input(
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_type_cost:"
                + fuel_type,
                val=DEFAULT_FUEL_UNIT_COST[fuel_type],
                units="USD/kg",
                desc="Amount of fuel from that tank which will be consumed during mission",
            )

        self.add_output(
            name="data:cost:fuel_cost",
            val=0.0,
            units="USD",
            desc="Fuel cost for single flight mission",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            outputs["data:cost:fuel_cost"] += (
                inputs[
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_type_cost:"
                    + fuel_type
                ]
                * inputs[
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_mission"
                ]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            partials[
                "data:cost:fuel_cost",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_consumed_mission",
            ] = inputs[
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_type_cost:"
                + fuel_type
            ]

            partials[
                "data:cost:fuel_cost",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_type_cost:"
                + fuel_type,
            ] += inputs[
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_consumed_mission"
            ]
