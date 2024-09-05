# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from .perf_inflight_emissions_sum import PerformancesBatteryPackInFlightEmissionsSum


class PerformancesBatteryPackInFlightEmissions(om.Group):
    """
    Emissions of all pollutants at each step of the flight.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        # No in-flight emissions for the battery pack
        ivc_emissions = om.IndepVarComp()
        ivc_emissions.add_output("CO2_emissions", units="g", val=np.zeros(number_of_points))
        ivc_emissions.add_output("CO_emissions", units="g", val=np.zeros(number_of_points))
        ivc_emissions.add_output("NOx_emissions", units="g", val=np.zeros(number_of_points))
        ivc_emissions.add_output("SOx_emissions", units="g", val=np.zeros(number_of_points))
        ivc_emissions.add_output("H2O_emissions", units="g", val=np.zeros(number_of_points))
        ivc_emissions.add_output("HC_emissions", units="g", val=np.zeros(number_of_points))

        self.add_subsystem(
            name="emissions",
            subsys=ivc_emissions,
            promotes=["*"],
        )
        self.add_subsystem(
            name="emissions_sum",
            subsys=PerformancesBatteryPackInFlightEmissionsSum(
                battery_pack_id=battery_pack_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
