# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_rpm_in import PerformancesRPMIn
from .perf_shaft_power_in import PerformancesShaftPowerIn
from .perf_torque_in import PerformancesTorqueIn
from .perf_torque_out import PerformancesTorqueOut
from .perf_maximum import PerformancesMaximum


class PerformancesSpeedReducer(om.Group):
    def initialize(self):
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        speed_reducer_id = self.options["speed_reducer_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "rpm_in",
            PerformancesRPMIn(number_of_points=number_of_points, speed_reducer_id=speed_reducer_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "shaft_power_in",
            PerformancesShaftPowerIn(
                number_of_points=number_of_points, speed_reducer_id=speed_reducer_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "torque_in",
            PerformancesTorqueIn(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "torque_out",
            PerformancesTorqueOut(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(
                number_of_points=number_of_points, speed_reducer_id=speed_reducer_id
            ),
            promotes=["*"],
        )
