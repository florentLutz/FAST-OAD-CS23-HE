# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_electric_node import PerformancesElectricalNode
from .perf_maximum import PerformancesMaximum


class PerformancesDCBus(om.Group):
    def initialize(self):
        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="number_of_inputs",
            default=1,
            types=int,
            desc="Number of connections at the input of the bus",
            allow_none=False,
        )
        self.options.declare(
            name="number_of_outputs",
            default=1,
            types=int,
            desc="Number of connections at the output of the bus",
            allow_none=False,
        )

    def setup(self):
        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]
        number_of_inputs = self.options["number_of_inputs"]
        number_of_outputs = self.options["number_of_outputs"]

        self.add_subsystem(
            name="electrical_node",
            subsys=PerformancesElectricalNode(
                number_of_points=number_of_points,
                number_of_inputs=number_of_inputs,
                number_of_outputs=number_of_outputs,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="maximum",
            subsys=PerformancesMaximum(
                dc_bus_id=dc_bus_id,
                number_of_points=number_of_points,
                number_of_inputs=number_of_inputs,
            ),
            promotes=["*"],
        )
