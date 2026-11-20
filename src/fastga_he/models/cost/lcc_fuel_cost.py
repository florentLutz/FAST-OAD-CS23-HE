# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging
import numpy as np
import openmdao.api as om

DEFAULT_FUEL_UNIT_COST = {"jet_fuel": 2.374, "diesel": 1.977, "avgas": 2.787, "hydrogen": 6.94}

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

        known_types = set(fuel_types) & set(DEFAULT_FUEL_UNIT_COST)
        has_unknown_fuel = any(f not in DEFAULT_FUEL_UNIT_COST for f in fuel_types)

        # Unknown fuel types are costed as jet fuel
        priced_types = known_types | ({"jet_fuel"} if has_unknown_fuel else set())

        for fuel_type in priced_types:
            self.add_subsystem(
                fuel_type + "_consumption",
                _FuelConsumption(
                    tank_types=tank_types,
                    tank_names=tank_names,
                    fuel_types=fuel_types,
                    fuel_type=fuel_type,
                ),
                promotes=["*"],
            )

        if priced_types:
            # Pass a copy so later set operations can't alter this subsystem's option
            self.add_subsystem("fuel_cost", _FuelCost(fuel_types=set(priced_types)), promotes=["*"])

        # Hydrocarbon fuels are every priced fuel except hydrogen
        hydrocarbon_fuel_types = priced_types - {"hydrogen"}

        if hydrocarbon_fuel_types:
            self.add_subsystem(
                "hydrocarbon_fuel_cost",
                _HydrocarbonFuelCost(fuel_types=hydrocarbon_fuel_types),
                promotes=["*"],
            )

        self.add_subsystem(
            "total_fuel_cost",
            _TotalFuelCost(),
            promotes=["*"],
        )


class _FuelConsumption(om.ExplicitComponent):
    """
    Computation of the fuel consumption for a specific fuel type per mission.
    """

    def initialize(self):
        self.options.declare("tank_types", types=list, default=[])
        self.options.declare("tank_names", types=list, default=[])
        self.options.declare("fuel_types", types=list, default=[])
        self.options.declare("fuel_type", types=str, default="jet_fuel")

    def setup(self):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]
        given_fuel_type = self.options["fuel_type"]

        for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
            self.add_input(
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_id
                + ":fuel_consumed_main_route",
                units="kg",
                val=np.nan,
            )

            if fuel_type not in DEFAULT_FUEL_UNIT_COST and given_fuel_type == "jet_fuel":
                _LOGGER.warning("Unregistered fuel type, replaced by Jet-A1!")

        self.add_output(
            name=given_fuel_type + "_consumption",
            val=0.0,
            units="kg",
        )

    def setup_partials(self):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]
        given_fuel_type = self.options["fuel_type"]

        if given_fuel_type == "jet_fuel":
            for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
                if fuel_type == given_fuel_type or fuel_type not in DEFAULT_FUEL_UNIT_COST:
                    self.declare_partials(
                        of=given_fuel_type + "_consumption",
                        wrt="data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route",
                        val=1.0,
                    )

                else:
                    self.declare_partials(
                        of=given_fuel_type + "_consumption",
                        wrt="data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route",
                        val=0.0,
                    )

        else:
            for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
                if fuel_type == given_fuel_type:
                    self.declare_partials(
                        of=given_fuel_type + "_consumption",
                        wrt="data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route",
                        val=1.0,
                    )

                else:
                    self.declare_partials(
                        of=given_fuel_type + "_consumption",
                        wrt="data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route",
                        val=0.0,
                    )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tank_types = self.options["tank_types"]
        tank_names = self.options["tank_names"]
        fuel_types = self.options["fuel_types"]
        given_fuel_type = self.options["fuel_type"]

        fuel_consumption = 0.0

        if given_fuel_type == "jet_fuel":
            for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
                if fuel_type == given_fuel_type or fuel_type not in DEFAULT_FUEL_UNIT_COST:
                    fuel_consumption += inputs[
                        "data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route"
                    ]

        else:
            for tank_type, tank_id, fuel_type in zip(tank_types, tank_names, fuel_types):
                if fuel_type == given_fuel_type:
                    fuel_consumption += inputs[
                        "data:propulsion:he_power_train:"
                        + tank_type
                        + ":"
                        + tank_id
                        + ":fuel_consumed_main_route"
                    ]

        outputs[given_fuel_type + "_consumption"] = fuel_consumption


class _FuelCost(om.ExplicitComponent):
    """
    Computation of the hydrocarbon fuel cost of the aircraft for single mission. The unit price of
    avgas 100LL and Jet-A1 are obtained from
    https://orleans.aeroport.fr/aeroport-dorleans-loire-valley-lfoz/. The cost of unit
    hydrogen is obtained from :cite:`sens:2024`. The reference unit price of hydrogen is obtained
    from https://www.energymarketprice.com/home/en/news/1172424
    """

    def initialize(self):
        self.options.declare("fuel_types", types=set, default=set())

    def setup(self):
        fuel_types = self.options["fuel_types"]

        for fuel_type in fuel_types:
            if fuel_type == "hydrogen":
                self.add_input(
                    "hydrogen_consumption",
                    units="kg",
                    val=np.nan,
                )
                self.add_input(
                    "data:cost:hydrogen_per_unit_mass_cost",
                    val=DEFAULT_FUEL_UNIT_COST[fuel_type],
                    units="USD/kg",
                    desc=fuel_type + "cost per unit mass",
                )

                self.add_output(
                    name="data:cost:hydrogen_fuel_cost",
                    val=0.0,
                    units="USD",
                )

            else:
                self.add_input(
                    fuel_type + "_consumption",
                    units="kg",
                    val=np.nan,
                )
                self.add_input(
                    "data:cost:" + fuel_type + "_per_unit_mass_cost",
                    val=DEFAULT_FUEL_UNIT_COST[fuel_type],
                    units="USD/kg",
                    desc=fuel_type + "cost per unit mass",
                )

                self.add_output(
                    name="data:cost:" + fuel_type + "_cost",
                    val=0.0,
                    units="USD",
                )

    def setup_partials(self):
        fuel_types = self.options["fuel_types"]

        for fuel_type in fuel_types:
            if fuel_type == "hydrogen":
                self.declare_partials(
                    of="data:cost:hydrogen_fuel_cost",
                    wrt=[
                        "data:cost:hydrogen_per_unit_mass_cost",
                        "hydrogen_consumption",
                    ],
                    method="exact",
                )

            else:
                self.declare_partials(
                    of="data:cost:" + fuel_type + "_cost",
                    wrt=[
                        "data:cost:" + fuel_type + "_per_unit_mass_cost",
                        fuel_type + "_consumption",
                    ],
                    method="exact",
                )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_types = self.options["fuel_types"]

        for fuel_type in fuel_types:
            if fuel_type == "hydrogen":
                outputs["data:cost:hydrogen_fuel_cost"] = (
                    inputs["data:cost:hydrogen_per_unit_mass_cost"] * inputs["hydrogen_consumption"]
                )

            else:
                outputs["data:cost:" + fuel_type + "_cost"] = (
                    inputs["data:cost:" + fuel_type + "_per_unit_mass_cost"]
                    * inputs[fuel_type + "_consumption"]
                )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_types = self.options["fuel_types"]

        for fuel_type in fuel_types:
            if fuel_type == "hydrogen":
                partials[
                    "data:cost:hydrogen_fuel_cost",
                    "data:cost:hydrogen_per_unit_mass_cost",
                ] = inputs["hydrogen_consumption"]

                partials["data:cost:hydrogen_fuel_cost", "hydrogen_consumption"] = inputs[
                    "data:cost:hydrogen_per_unit_mass_cost"
                ]

            else:
                partials[
                    "data:cost:" + fuel_type + "_cost",
                    "data:cost:" + fuel_type + "_per_unit_mass_cost",
                ] = inputs[fuel_type + "_consumption"]

                partials["data:cost:" + fuel_type + "_cost", fuel_type + "_consumption"] = inputs[
                    "data:cost:" + fuel_type + "_per_unit_mass_cost"
                ]


class _HydrocarbonFuelCost(om.ExplicitComponent):
    """
    Computation of the hydrocarbon fuel cost of the aircraft for single mission.
    """

    def initialize(self):
        self.options.declare("fuel_types", types=set, default=set())

    def setup(self):
        fuel_types = self.options["fuel_types"]

        for fuel_type in fuel_types:
            self.add_input(
                name="data:cost:" + fuel_type + "_cost",
                val=np.nan,
                units="USD",
            )

        self.add_output(
            name="data:cost:hydrocarbon_fuel_cost",
            val=0.0,
            units="USD",
            desc="hydrocarbon Fuel cost for single flight mission",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_types = self.options["fuel_types"]

        outputs["data:cost:hydrocarbon_fuel_cost"] = sum(
            inputs["data:cost:" + fuel_type + "_cost"] for fuel_type in fuel_types
        )


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
