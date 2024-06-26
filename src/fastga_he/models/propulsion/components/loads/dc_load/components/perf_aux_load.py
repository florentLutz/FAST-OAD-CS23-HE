# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_power_in import PerformancesPowerIn
from .perf_current_in import PerformancesCurrentIn
from .perf_maximum import PerformancesMaximum


class PerformancesAuxLoad(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):

        aux_load_id = self.options["aux_load_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "power_in_mission",
            PerformancesPowerIn(aux_load_id=aux_load_id, number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "current_in",
            PerformancesCurrentIn(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(aux_load_id=aux_load_id, number_of_points=number_of_points),
            promotes=["*"],
        )
