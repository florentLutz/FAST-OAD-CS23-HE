# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_fuel_output import PerformancesFuelOutput
from .perf_fuel_input import PerformancesFuelInput
from .perf_total_fuel_flowed import PerformancesTotalFuelFlowed


class PerformancesFuelSystem(om.Group):
    """
    Group that gathers all the components necessary to assess the performances of the fuel system.
    """

    def initialize(self):

        self.options.declare(
            name="fuel_system_id",
            default=None,
            desc="Identifier of the fuel system",
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
            desc="Number of connections at the input of the fuel system, should always be tanks",
            allow_none=False,
        )
        self.options.declare(
            name="number_of_engines",
            default=1,
            types=int,
            desc="Number of connections at the output of the fuel system, should always be engine",
            allow_none=False,
        )

    def setup(self):

        fuel_system_id = self.options["fuel_system_id"]
        number_of_points = self.options["number_of_points"]
        number_of_tanks = self.options["number_of_tanks"]
        number_of_engines = self.options["number_of_engines"]

        self.add_subsystem(
            name="fuel_flow_out",
            subsys=PerformancesFuelOutput(
                number_of_points=number_of_points,
                number_of_engines=number_of_engines,
                fuel_system_id=fuel_system_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fuel_flow_in",
            subsys=PerformancesFuelInput(
                number_of_points=number_of_points,
                number_of_tanks=number_of_tanks,
                fuel_system_id=fuel_system_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="total_fuel_flowed",
            subsys=PerformancesTotalFuelFlowed(
                number_of_points=number_of_points, fuel_system_id=fuel_system_id
            ),
            promotes=["*"],
        )

        # Because I don't want to have to give the option on the number of engine to the sizing
        # group, I'll make it an output of one of those component. It was initially anivc but since ivc output appear as input of the problem (???) I have to do it someway else
