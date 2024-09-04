# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .perf_density_ratio import PerformancesDensityRatio
from .perf_mach import PerformancesMach
from .perf_required_power import PerformancesRequiredPower
from .perf_fuel_consumption import PerformancesTurboshaftFuelConsumption
from .perf_fuel_consumed import PerformancesTurboshaftFuelConsumed
from .perf_sfc import PerformancesSFC
from .perf_power_for_power_rate import PerformancesPowerForPowerRate
from .perf_max_power_itt_limit import PerformancesMaxPowerITTLimit
from .perf_equivalent_rated_power_itt_limit import PerformancesEquivalentRatedPowerITTLimit
from .perf_max_power_opr_limit import PerformancesMaxPowerOPRLimit
from .perf_equivalent_rated_power_opr_limit import PerformancesEquivalentRatedPowerOPRLimit
from .perf_maximum import PerformancesMaximum


class PerformancesTurboshaft(om.Group):
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
            name="density_ratio",
            subsys=PerformancesDensityRatio(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="mach",
            subsys=PerformancesMach(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_required",
            subsys=PerformancesRequiredPower(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_for_power_rate",
            subsys=PerformancesPowerForPowerRate(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fuel_consumption",
            subsys=PerformancesTurboshaftFuelConsumption(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="fuel_consumed",
            subsys=PerformancesTurboshaftFuelConsumed(number_of_points=number_of_points),
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
        self.add_subsystem(
            name="sfc",
            subsys=PerformancesSFC(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="max_power_itt_limit",
            subsys=PerformancesMaxPowerITTLimit(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="equivalent_power_itt_limit",
            subsys=PerformancesEquivalentRatedPowerITTLimit(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="max_power_opr_limit",
            subsys=PerformancesMaxPowerOPRLimit(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="equivalent_power_opr_limit",
            subsys=PerformancesEquivalentRatedPowerOPRLimit(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="maximum",
            subsys=PerformancesMaximum(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
