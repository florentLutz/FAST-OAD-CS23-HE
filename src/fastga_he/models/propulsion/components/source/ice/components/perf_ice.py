# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..components.perf_torque import PerformancesTorque
from ..components.perf_equivalent_sl_power import PerformancesEquivalentSeaLevelPower
from ..components.perf_mean_effective_pressure import PerformancesMeanEffectivePressure
from ..components.perf_sfc import PerformancesSFC
from ..components.perf_fuel_consumption import PerformancesICEFuelConsumption
from ..components.perf_fuel_consumed import PerformancesICEFuelConsumed
from ..components.perf_maximum import PerformancesMaximum

from ..components.perf_inflight_emissions import PerformancesICEInFlightEmissions


class PerformancesICE(om.Group):
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
        ice_id = self.options["ice_id"]
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
                number_of_points=number_of_points, ice_id=ice_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="sfc",
            subsys=PerformancesSFC(number_of_points=number_of_points, ice_id="ice_1"),
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
            subsys=PerformancesICEInFlightEmissions(
                number_of_points=number_of_points, ice_id=ice_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="maximum",
            subsys=PerformancesMaximum(number_of_points=number_of_points, ice_id=ice_id),
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
