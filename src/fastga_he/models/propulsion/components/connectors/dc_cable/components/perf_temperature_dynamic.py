# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from .perf_temperature_derivative import PerformancesTemperatureDerivative
from .perf_temperature_increase import PerformancesTemperatureIncrease
from .perf_temperature_from_increase import PerformancesTemperatureFromIncrease

from ..constants import SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE


@oad.RegisterSubmodel(
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE,
    "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.with_dynamics",
)
class PerformancesTemperatureDynamics(om.Group):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        harness_id = self.options["harness_id"]

        self.add_subsystem(
            "temperature_derivative",
            PerformancesTemperatureDerivative(
                number_of_points=number_of_points, harness_id=harness_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_increase",
            PerformancesTemperatureIncrease(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_from_increase",
            PerformancesTemperatureFromIncrease(
                number_of_points=number_of_points, harness_id=harness_id
            ),
            promotes=["*"],
        )
