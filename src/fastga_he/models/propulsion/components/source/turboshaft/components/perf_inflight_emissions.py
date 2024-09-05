# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_inflight_co2_emissions import PerformancesTurboshaftInFlightCO2Emissions
from .perf_inflight_co_emissions import PerformancesTurboshaftInFlightCOEmissions
from .perf_inflight_nox_emissions import PerformancesTurboshaftInFlightNOxEmissions
from .perf_inflight_sox_emissions import PerformancesTurboshaftInFlightSOxEmissions
from .perf_inflight_h2o_emissions import PerformancesTurboshaftInFlightH2OEmissions
from .perf_inflight_hc_emissions import PerformancesTurboshaftInFlightHCEmissions
from .perf_inflight_emissions_sum import PerformancesTurboshaftInFlightEmissionsSum


class PerformancesTurboshaftInFlightEmissions(om.Group):
    """
    Addition of the emissions of all pollutants at each step of the flight.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        self.add_subsystem(
            name="CO2_emissions",
            subsys=PerformancesTurboshaftInFlightCO2Emissions(
                turboshaft_id=turboshaft_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="CO_emissions",
            subsys=PerformancesTurboshaftInFlightCOEmissions(
                turboshaft_id=turboshaft_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="NOx_emissions",
            subsys=PerformancesTurboshaftInFlightNOxEmissions(
                turboshaft_id=turboshaft_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="SOx_emissions",
            subsys=PerformancesTurboshaftInFlightSOxEmissions(
                turboshaft_id=turboshaft_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="water_vapour_emissions",
            subsys=PerformancesTurboshaftInFlightH2OEmissions(
                turboshaft_id=turboshaft_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="HC_emissions",
            subsys=PerformancesTurboshaftInFlightHCEmissions(
                turboshaft_id=turboshaft_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="emissions_sum",
            subsys=PerformancesTurboshaftInFlightEmissionsSum(
                turboshaft_id=turboshaft_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
