# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_fuel_mission_consumed import PerformancesHydrogenGasConsumedMission
from ..components.perf_fuel_remaining import PerformancesHydrogenGasRemainingMission


class PerformancesHydrogenGasTank(om.Group):
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
            name="hydrogen_gas_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        self.add_subsystem(
            "hydrogen_gas_consumed_mission",
            PerformancesHydrogenGasConsumedMission(
                number_of_points=number_of_points, hydrogen_gas_tank_id=hydrogen_gas_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "hydrogen_gas_remaining_mission",
            PerformancesHydrogenGasRemainingMission(
                number_of_points=number_of_points, hydrogen_gas_tank_id=hydrogen_gas_tank_id
            ),
            promotes=["*"],
        )
