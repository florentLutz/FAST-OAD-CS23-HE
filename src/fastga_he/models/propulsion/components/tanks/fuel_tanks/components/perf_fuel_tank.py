# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_fuel_consumed_mission import PerformancesFuelConsumedMission
from ..components.perf_fuel_consumed_main_route import PerformancesFuelConsumedMainRoute
from ..components.perf_fuel_remaining import PerformancesFuelRemainingMission


class PerformancesFuelTank(om.Group):
    """
    Regrouping all the components for the performances of the tank. Note that to limit the work
    to be done for the implementation of fuel tanks, fuel tanks don't output the fuel consumed
    used to iterate on the mass during the mission, but it uses it. Just like for the CG where we
    will output the "varying" part of the CG straight from the mission; we could do the same for
    mass (which may or may not improve the computation time).
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "number_of_points_reserve",
            default=None,  # By default, for retro-compatibility, we want it disabled
            desc="number of equilibrium to be treated in reserve",
            types=int,
        )
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        self.add_subsystem(
            "fuel_consumed_mission",
            PerformancesFuelConsumedMission(
                number_of_points=self.options["number_of_points"],
                fuel_tank_id=self.options["fuel_tank_id"],
            ),
            promotes=["*"],
        )

        if self.options["number_of_points_reserve"]:
            self.add_subsystem(
                "fuel_consumed_main_route",
                PerformancesFuelConsumedMainRoute(
                    number_of_points=self.options["number_of_points"],
                    fuel_tank_id=self.options["fuel_tank_id"],
                    number_of_points_reserve=self.options["number_of_points_reserve"],
                ),
                promotes=["*"],
            )

        self.add_subsystem(
            "fuel_remaining_mission",
            PerformancesFuelRemainingMission(
                number_of_points=self.options["number_of_points"],
                fuel_tank_id=self.options["fuel_tank_id"],
            ),
            promotes=["*"],
        )
