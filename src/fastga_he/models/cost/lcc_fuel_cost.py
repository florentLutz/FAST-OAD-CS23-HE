# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging
import numpy as np
import openmdao.api as om

DEFAULT_FUEL_UNIT_COST = {"jet_fuel": 2.967, "diesel": 1.977, "avgas": 3.66, "hydrogen": 6.94}

_LOGGER = logging.getLogger(__name__)


class LCCFuelCost(om.Group):
    """
    Group to compute the fuel cost of the aircraft for single mission.
    """

    def initialize(self):
        self.options.declare("tank_types", types=list, default=[])
        self.options.declare("tank_names", types=list, default=[])
        self.options.declare("fuel_types", types=list, default=[])

    def setup(self):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        if "hydrogen" in fuel_types:
            self.add_subsystem(
                "hydrogen_fuel_cost",
                _HydrogenFuelCost(
                    tank_types=tank_types, tank_names=tank_names, fuel_types=fuel_types
                ),
                promotes=["*"],
            )

        if any(fuel_type != "hydrogen" for fuel_type in fuel_types):
            self.add_subsystem(
                "hydrocarbon_fuel_cost",
                _HydrocarbonFuelCost(
                    tank_types=tank_types, tank_names=tank_names, fuel_types=fuel_types
                ),
                promotes=["*"],
            )

        self.add_subsystem(
            "total_fuel_cost",
            _TotalFuelCost(),
            promotes=["*"],
        )


class _HydrogenFuelCost(om.ExplicitComponent):
    """
    Computation of the fuel cost of the aircraft for single mission. The cost of unit hydrogen is
    obtained from :cite:`sens:2024`. The reference unit price of hydrogen is obtained from
    https://www.energymarketprice.com/home/en/news/1172424
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
            if fuel_type == "hydrogen":
                self.add_input(
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_main_route",
                    units="kg",
                    val=np.nan,
                    desc="Amount of fuel from that tank which will be consumed during mission",
                )
                self.add_input(
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_type_cost:"
                    + fuel_type,
                    val=DEFAULT_FUEL_UNIT_COST[fuel_type],
                    units="USD/kg",
                    desc="Amount of fuel from that tank which will be consumed during main route",
                )

                self.declare_partials(
                    of="*",
                    wrt=[
                        "data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route",
                        "data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_type_cost:"
                        + fuel_type,
                    ],
                    method="exact",
                )

        self.add_output(
            name="data:cost:hydrogen_fuel_cost",
            val=0.0,
            units="USD",
            desc="Hydrogen Fuel cost for single flight mission",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        hydrogen_fuel_cost = 0.0

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            if fuel_type == "hydrogen":
                hydrogen_fuel_cost += (
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
                        + ":fuel_consumed_main_route"
                    ]
                )

        outputs["data:cost:hydrogen_fuel_cost"] = hydrogen_fuel_cost

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            if fuel_type == "hydrogen":
                partials[
                    "data:cost:hydrogen_fuel_cost",
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_main_route",
                ] = inputs[
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_type_cost:"
                    + fuel_type
                ]

                partials[
                    "data:cost:hydrogen_fuel_cost",
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_type_cost:"
                    + fuel_type,
                ] = inputs[
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_main_route"
                ]


class _HydrocarbonFuelCost(om.ExplicitComponent):
    """
    Computation of the hydrocarbon fuel cost of the aircraft for single mission. The unit price of
    avgas 100LL and Jet-A1 are obtained from https://orleans.aeroport.fr.
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
            if fuel_type != "hydrogen":
                self.add_input(
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_main_route",
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
                    desc="Amount of fuel from that tank which will be consumed during main route",
                )

                self.declare_partials(
                    of="*",
                    wrt=[
                        "data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route",
                        "data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_type_cost:"
                        + fuel_type,
                    ],
                    method="exact",
                )

        self.add_output(
            name="data:cost:hydrocarbon_fuel_cost",
            val=0.0,
            units="USD",
            desc="Fossil Fuel cost for single flight mission",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        hydrocarbon_fuel_cost = 0.0

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            if fuel_type != "hydrogen":
                hydrocarbon_fuel_cost += (
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
                        + ":fuel_consumed_main_route"
                    ]
                )

        outputs["data:cost:hydrocarbon_fuel_cost"] = hydrocarbon_fuel_cost

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            if fuel_type != "hydrogen":
                partials[
                    "data:cost:hydrocarbon_fuel_cost",
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_main_route",
                ] = inputs[
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_type_cost:"
                    + fuel_type
                ]

                partials[
                    "data:cost:hydrocarbon_fuel_cost",
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_type_cost:"
                    + fuel_type,
                ] = inputs[
                    "data:propulsion:he_power_train:"
                    + tank_type
                    + ":"
                    + tank_id
                    + ":fuel_consumed_main_route"
                ]


class _TotalFuelCost(om.ExplicitComponent):
    """
    Computation of the total  fuel cost of the aircraft for single mission.
    """

    def setup(self):
        self.add_input(
            "data:cost:hydrocarbon_fuel_cost",
            units="USD",
            val=0.0,
            desc="Fossil Fuel cost for single flight mission",
        )
        self.add_input(
            "data:cost:hydrogen_fuel_cost",
            units="USD",
            val=0.0,
            desc="Hydrogen Fuel cost for single flight mission",
        )

        self.add_output(
            name="data:cost:total_fuel_cost",
            val=0.0,
            units="USD",
            desc="Total fuel cost for single flight mission",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:total_fuel_cost"] = (
            inputs["data:cost:hydrocarbon_fuel_cost"] + inputs["data:cost:hydrogen_fuel_cost"]
        )
