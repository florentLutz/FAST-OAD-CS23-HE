# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_humidifier_oxidizer_temperature import PerformancesHumidifierOxidizerTemperature
from .perf_humidifier_oxidizer_pressure import PerformancesHumidifierOxidizerPressure
from .perf_humidifier_air_pressure_drop import PerformancesHumidifierRatingPressureDrop

from ..fluid_characteristics import FluidDensity


class PerformancesHumidifier(om.Group):
    """
    Maximum pressure drop computation of the humidifier during mission.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="humidifier_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["humidifier_id"]

        self.add_subsystem(
            "humidifier_oxidizer_temperature",
            PerformancesHumidifierOxidizerTemperature(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "humidifier_oxidizer_pressure",
            PerformancesHumidifierOxidizerPressure(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "compressed_air_density",
            FluidDensity(number_of_points=number_of_points),
            promotes=[
                ("fluid_pressure", "oxidizer_pressure"),
                ("fluid_temperature", "oxidizer_temperature"),
                ("fluid_density", "humidifier_air_density"),
            ],
        )
        self.add_subsystem(
            "humidifier_rating_pressure_drop",
            PerformancesHumidifierRatingPressureDrop(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                humidifier_id=humidifier_id,
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
