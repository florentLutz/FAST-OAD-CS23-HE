# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_design_air_flow import PerformancesInletDesignAirFlow
from .perf_air_dynamic_pressure import PerformancesAirDynamicPressure
from .perf_throat_airspeed import PerformancesThroatAirSpeed
from .perf_ambient_total_pressure import PerformancesAmbientTotalPressure
from .perf_inlet_air_mass_flow import PerformancesAirInletAirMassFlow
from .perf_design_flow_area import PerformancesInletDesignFlowArea
from .perf_inlet_design_dynamic_pressure import PerformancesInletDesignDynamicPressure

from ......loads.sm_pmsm.components.perf_air_dynamic_viscosity import (
    PerformancesAirDynamicViscosity,
)
from ...perf_ambient_pressure import PerformancesPEMFCStackBOPAmbientPressure


class PerformancesScoopInlet(om.Group):
    """
    Air inlet Performances group.
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
            desc="Identifier of the air inlet",
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
            "inlet_throat_airspeed",
            PerformancesThroatAirSpeed(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_design_flow_area",
            PerformancesInletDesignFlowArea(
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
            "inlet_design_dynamic_pressure",
            PerformancesInletDesignDynamicPressure(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
