# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_conduction_loss import PerformancesConductionLosses
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_total_loss import PerformancesLosses


class PerformanceInverter(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "conduction_losses",
            PerformancesConductionLosses(
                inverter_id=inverter_id, number_of_points=number_of_points
            ),
            promotes=["data:*", "current", "modulation_index"],
        )
        self.add_subsystem(
            "switching_losses",
            PerformancesSwitchingLosses(inverter_id=inverter_id, number_of_points=number_of_points),
            promotes=["data:*", "current", "switching_frequency"],
        )
        self.add_subsystem(
            "total_losses",
            PerformancesLosses(number_of_points=number_of_points),
            promotes=["losses_inverter"],
        )

        self.connect(
            "conduction_losses.conduction_losses_diode", "total_losses.conduction_losses_diode"
        )
        self.connect(
            "conduction_losses.conduction_losses_IGBT", "total_losses.conduction_losses_IGBT"
        )
        self.connect(
            "switching_losses.switching_losses_diode", "total_losses.switching_losses_diode"
        )
        self.connect("switching_losses.switching_losses_IGBT", "total_losses.switching_losses_IGBT")
