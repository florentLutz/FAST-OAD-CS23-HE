# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .perf_inflight_co2_emissions import PerformancesHighRPMICEInFlightCO2Emissions
from .perf_inflight_co_emissions import PerformancesHighRPMICEInFlightCOEmissions
from .perf_inflight_nox_emissions import PerformancesHighRPMICEInFlightNOxEmissions
from .perf_inflight_sox_emissions import PerformancesHighRPMICEInFlightSOxEmissions
from .perf_inflight_h2o_emissions import PerformancesHighRPMICEInFlightH2OEmissions
from .perf_inflight_hc_emissions import PerformancesHighRPMICEInFlightHCEmissions
from .perf_inflight_lead_emissions import PerformancesHighRPMICEInFlightLeadEmissions
from .perf_inflight_emissions_sum import PerformancesHighRPMICEInFlightEmissionsSum


class PerformancesHighRPMICEInFlightEmissions(om.Group):
    """
    Addition of the emissions of all pollutants at each step of the flight.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        self.add_subsystem(
            name="CO2_emissions",
            subsys=PerformancesHighRPMICEInFlightCO2Emissions(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="CO_emissions",
            subsys=PerformancesHighRPMICEInFlightCOEmissions(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="NOx_emissions",
            subsys=PerformancesHighRPMICEInFlightNOxEmissions(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="SOx_emissions",
            subsys=PerformancesHighRPMICEInFlightSOxEmissions(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="water_vapour_emissions",
            subsys=PerformancesHighRPMICEInFlightH2OEmissions(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="HC_emissions",
            subsys=PerformancesHighRPMICEInFlightHCEmissions(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="lead_emissions",
            subsys=PerformancesHighRPMICEInFlightLeadEmissions(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="emissions_sum",
            subsys=PerformancesHighRPMICEInFlightEmissionsSum(
                high_rpm_ice_id=high_rpm_ice_id, number_of_points=number_of_points
            ),
            promotes=["*"],
        )
