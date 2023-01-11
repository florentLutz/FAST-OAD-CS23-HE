# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_mission_rpm import PerformancesRPMMission
from .perf_advance_ratio import PerformancesAdvanceRatio
from .perf_tip_mach import PerformancesTipMach
from .perf_blade_reynolds import PerformancesBladeReynoldsNumber
from .perf_thrust_coefficient import PerformancesThrustCoefficient
from .perf_power_coefficient import PerformancesPowerCoefficient
from .perf_efficiency import PerformancesEfficiency
from .perf_shaft_power import PerformancesShaftPower
from .perf_torque import PerformancesTorque
from .perf_maximum import PerformancesMaximum


class PerformancesPropeller(om.Group):
    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "rpm_mission",
            PerformancesRPMMission(propeller_id=propeller_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "advance_ratio",
            PerformancesAdvanceRatio(propeller_id=propeller_id, number_of_points=number_of_points),
            promotes=["true_airspeed", "rpm", "data:*", "advance_ratio"],
        )
        self.add_subsystem(
            "tip_mach",
            PerformancesTipMach(propeller_id=propeller_id, number_of_points=number_of_points),
            promotes=["true_airspeed", "rpm", "altitude", "data:*"],
        )
        self.add_subsystem(
            "blade_diameter_reynolds",
            PerformancesBladeReynoldsNumber(
                propeller_id=propeller_id, number_of_points=number_of_points
            ),
            promotes=["true_airspeed", "rpm", "altitude", "data:*"],
        )

        self.add_subsystem(
            "thrust_coefficient",
            PerformancesThrustCoefficient(
                propeller_id=propeller_id, number_of_points=number_of_points
            ),
            promotes=["rpm", "thrust", "altitude", "data:*"],
        )
        self.add_subsystem(
            "power_coefficient",
            PerformancesPowerCoefficient(
                propeller_id=propeller_id, number_of_points=number_of_points
            ),
            promotes=["data:*", "advance_ratio"],
        )
        self.add_subsystem(
            "efficiency",
            PerformancesEfficiency(number_of_points=number_of_points),
            promotes=["efficiency", "advance_ratio"],
        )
        self.add_subsystem(
            "shaft_power",
            PerformancesShaftPower(propeller_id=propeller_id, number_of_points=number_of_points),
            promotes=["data:*", "altitude", "shaft_power_in", "rpm"],
        )

        self.add_subsystem(
            "torque",
            PerformancesTorque(number_of_points=number_of_points),
            promotes=["shaft_power_in", "rpm", "torque_in"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(propeller_id=propeller_id, number_of_points=number_of_points),
            promotes=["data:*", "rpm", "torque_in", "advance_ratio"],
        )

        self.connect("tip_mach.tip_mach", ["power_coefficient.tip_mach", "maximum.tip_mach"])
        self.connect("blade_diameter_reynolds.reynolds_D", "power_coefficient.reynolds_D")
        self.connect(
            "thrust_coefficient.thrust_coefficient",
            ["power_coefficient.thrust_coefficient", "efficiency.thrust_coefficient"],
        )
        self.connect(
            "power_coefficient.power_coefficient",
            ["shaft_power.power_coefficient", "efficiency.power_coefficient"],
        )
