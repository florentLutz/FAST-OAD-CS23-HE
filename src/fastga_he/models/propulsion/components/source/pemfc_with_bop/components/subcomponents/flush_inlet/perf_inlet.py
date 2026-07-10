# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_design_air_flow import PerformancesInletDesignAirFlow
from .perf_inlet_drag import PerformancesInletDrag
from .perf_air_dynamic_pressure import PerformancesAirDynamicPressure
from .perf_max_ramp_pressure_efficiency import PerformancesMaxRamPressureEfficiency
from .perf_pressure_efficiency_difference_factor import (
    PerformancesPressureEfficiencyDifferenceFactor,
)
from .perf_pressure_efficiency_difference import PerformancesPressureEfficiencyDifference
from .perf_inlet_efficiency import PerformancesInletEfficiency
from .perf_throat_airspeed import PerformancesThroatAirSpeed
from .perf_ambient_total_pressure import PerformancesAmbientTotalPressure
from .perf_throat_total_temperature import PerformancesThroatTemperature
from .perf_throat_total_pressure import PerformancesThroatPressure
from .perf_inlet_pressure_drop import PerformancesInletPressureDrop
from .perf_inlet_air_mass_flow import PerformancesAirInletAirMassFlow

from ......loads.sm_pmsm.components.perf_air_dynamic_viscosity import (
    PerformancesAirDynamicViscosity,
)
from ...perf_ambient_pressure import PerformancesPEMFCStackBOPAmbientPressure


class PerformancesInlet(om.Group):
    """
    Air flush_inlet Performances group.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air flush_inlet",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "inlet_air_dynamic_pressure",
            PerformancesAirDynamicPressure(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_air_dynamic_viscosity",
            PerformancesAirDynamicViscosity(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_ambient_pressure",
            PerformancesPEMFCStackBOPAmbientPressure(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_air_flow",
            PerformancesAirInletAirMassFlow(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_design_air_flow",
            PerformancesInletDesignAirFlow(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_drag",
            PerformancesInletDrag(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "max_ramp_pressure_efficiency",
            PerformancesMaxRamPressureEfficiency(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_pressure_efficiency_difference_factor",
            PerformancesPressureEfficiencyDifferenceFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_pressure_efficiency_difference",
            PerformancesPressureEfficiencyDifference(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_efficiency",
            PerformancesInletEfficiency(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_throat_airspeed",
            PerformancesThroatAirSpeed(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_ambient_total_pressure",
            PerformancesAmbientTotalPressure(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_throat_total_temperature",
            PerformancesThroatTemperature(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_throat_total_pressure",
            PerformancesThroatPressure(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "max_inlet_pressure_drop",
            PerformancesInletPressureDrop(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
