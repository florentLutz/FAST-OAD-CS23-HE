# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_inflight_co2_emissions import PerformancesICEInFlightCO2Emissions
from .perf_inflight_co_emissions import PerformancesICEInFlightCOEmissions
from .perf_inflight_nox_emissions import PerformancesICEInFlightNOxEmissions
from .perf_inflight_sox_emissions import PerformancesICEInFlightSOxEmissions
from .perf_inflight_h2o_emissions import PerformancesICEInFlightH2OEmissions
from .perf_inflight_hc_emissions import PerformancesICEInFlightHCEmissions
from .perf_inflight_lead_emissions import PerformancesICEInFlightLeadEmissions
from .perf_inflight_emissions_sum import PerformancesICEInFlightEmissionsSum


class PerformancesICEInFlightEmissions(om.Group):
    """
    Addition of the emissions of all pollutants at each step of the flight.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        ice_id = self.options["ice_id"]

        self.add_subsystem(
            name="CO2_emissions",
            subsys=PerformancesICEInFlightCO2Emissions(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="CO_emissions",
            subsys=PerformancesICEInFlightCOEmissions(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="NOx_emissions",
            subsys=PerformancesICEInFlightNOxEmissions(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="SOx_emissions",
            subsys=PerformancesICEInFlightSOxEmissions(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="water_vapour_emissions",
            subsys=PerformancesICEInFlightH2OEmissions(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="HC_emissions",
            subsys=PerformancesICEInFlightHCEmissions(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="lead_emissions",
            subsys=PerformancesICEInFlightLeadEmissions(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="emissions_sum",
            subsys=PerformancesICEInFlightEmissionsSum(
                ice_id=ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
