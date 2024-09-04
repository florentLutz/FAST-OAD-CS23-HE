# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_current import PerformancesDCSSPCCurrent
from .perf_voltage_out import PerformancesDCSSPCVoltageOut
from .perf_maximum import PerformancesDCSSPCMaximum
from .perf_losses import PerformancesDCSSPCLosses
from .perf_power import PerformancesDCSSPCPower
from .perf_efficiency import PerformancesDCSSPCEfficiency


class PerformancesDCSSPC(om.Group):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )
        self.options.declare(
            "closed",
            default=True,
            desc="Boolean to choose whether the breaker is closed or not.",
            types=bool,
        )
        self.options.declare(
            "at_bus_output",
            default=True,
            desc="Boolean to inform whether the breaker is at the input or output of a bus.",
            types=bool,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        dc_sspc_id = self.options["dc_sspc_id"]
        closed = self.options["closed"]
        at_bus_output = self.options["at_bus_output"]

        self.add_subsystem(
            "efficiency",
            PerformancesDCSSPCEfficiency(
                number_of_points=number_of_points,
                closed=closed,
                dc_sspc_id=dc_sspc_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "current",
            PerformancesDCSSPCCurrent(number_of_points=number_of_points, closed=closed),
            promotes=["*"],
        )
        self.add_subsystem(
            "voltage",
            PerformancesDCSSPCVoltageOut(
                number_of_points=number_of_points,
                at_bus_output=at_bus_output,
                closed=closed,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "losses",
            PerformancesDCSSPCLosses(number_of_points=number_of_points, closed=closed),
            promotes=["*"],
        )
        self.add_subsystem(
            "power",
            PerformancesDCSSPCPower(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesDCSSPCMaximum(dc_sspc_id=dc_sspc_id, number_of_points=number_of_points),
            promotes=["*"],
        )
