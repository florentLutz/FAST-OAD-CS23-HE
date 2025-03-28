# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

# These models are exactly the same and doesn't require options so let's import it for consistency
from ...ice.components.perf_torque import PerformancesTorque
from ...ice.components.perf_equivalent_sl_power import PerformancesEquivalentSeaLevelPower
from ...ice.components.perf_fuel_consumption import PerformancesICEFuelConsumption
from ...ice.components.perf_fuel_consumed import PerformancesICEFuelConsumed

from .perf_mean_effective_pressure import PerformancesMeanEffectivePressure
from .perf_sfc import PerformancesSFC
from .perf_inflight_emissions import PerformancesHighRPMICEInFlightEmissions
from .perf_maximum import PerformancesMaximum


class PerformancesHighRPMICE(om.Group):
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
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            name="torque",
            subsys=PerformancesTorque(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="equivalent_SL_power",
            subsys=PerformancesEquivalentSeaLevelPower(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="mep",
            subsys=PerformancesMeanEffectivePressure(
                number_of_points=number_of_points, high_rpm_ice_id=high_rpm_ice_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="sfc",
            subsys=PerformancesSFC(number_of_points=number_of_points, high_rpm_ice_id="ice_1"),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fuel_consumption",
            subsys=PerformancesICEFuelConsumption(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fuel_consumed",
            subsys=PerformancesICEFuelConsumed(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            name="emissions",
            subsys=PerformancesHighRPMICEInFlightEmissions(
                number_of_points=number_of_points, high_rpm_ice_id=high_rpm_ice_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="maximum",
            subsys=PerformancesMaximum(
                number_of_points=number_of_points, high_rpm_ice_id=high_rpm_ice_id
            ),
            promotes=["*"],
        )

        energy_consumed = om.IndepVarComp()
        energy_consumed.add_output(
            "non_consumable_energy_t", np.full(number_of_points, 0.0), units="W*h"
        )
        self.add_subsystem(
            "energy_consumed",
            energy_consumed,
            promotes=["non_consumable_energy_t"],
        )
