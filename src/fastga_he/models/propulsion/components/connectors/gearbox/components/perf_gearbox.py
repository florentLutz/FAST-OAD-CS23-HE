# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_rpm_in import PerformancesRPMIn
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_torque_out import PerformancesTorqueOut
from ..components.perf_torque_in import PerformancesTorqueIn
from ..components.perf_maximum import PerformancesMaximum


class PerformancesGearbox(om.Group):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )

    def setup(self):
        gearbox_id = self.options["gearbox_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "rpm_in",
            PerformancesRPMIn(number_of_points=number_of_points, gearbox_id=gearbox_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "shaft_power_in",
            PerformancesShaftPowerIn(number_of_points=number_of_points, gearbox_id=gearbox_id),
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
            PerformancesMaximum(number_of_points=number_of_points, gearbox_id=gearbox_id),
            promotes=["*"],
        )
