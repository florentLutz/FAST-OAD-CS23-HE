# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

from .perf_h2_fuel_output import PerformancesH2FuelSystemOutput
from .perf_h2_fuel_input import PerformancesH2FuelSystemInput
from .perf_h2_total_fuel_flowed import PerformancesTotalH2FuelFlowed


class PerformancesH2FuelSystem(om.Group):
    """
    Group that gathers all the components necessary to assess the performances of the hydrogen
    fuel system.
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="number_of_tanks",
            default=1,
            types=int,
            desc="Number of connections at the input of the hydrogen fuel system, should always be tanks",
            allow_none=False,
        )
        self.options.declare(
            name="number_of_power_sources",
            default=1,
            types=int,
            desc="Number of connections at the output of the hydrogen fuel system",
            allow_none=False,
        )

        # The followong option(s) is/are dummy option(s) to ensure compatibility
        self.options.declare(
            name="wing_related",
            default=False,
            types=bool,
            desc="Option identifies weather the system reaches inside the wing or not",
        )
        self.options.declare(
            name="compact",
            default=False,
            types=bool,
            desc="Option identifies weather the system is installed compactly in one position",
        )

    def setup(self):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        number_of_points = self.options["number_of_points"]
        number_of_tanks = self.options["number_of_tanks"]
        number_of_power_sources = self.options["number_of_power_sources"]

        self.add_subsystem(
            name="h2_fuel_flow_out",
            subsys=PerformancesH2FuelSystemOutput(
                number_of_points=number_of_points,
                number_of_power_sources=number_of_power_sources,
                h2_fuel_system_id=h2_fuel_system_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="h2_fuel_flow_in",
            subsys=PerformancesH2FuelSystemInput(
                number_of_points=number_of_points,
                number_of_tanks=number_of_tanks,
                h2_fuel_system_id=h2_fuel_system_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="total_h2_fuel_flowed",
            subsys=PerformancesTotalH2FuelFlowed(
                number_of_points=number_of_points, h2_fuel_system_id=h2_fuel_system_id
            ),
            promotes=["*"],
        )
