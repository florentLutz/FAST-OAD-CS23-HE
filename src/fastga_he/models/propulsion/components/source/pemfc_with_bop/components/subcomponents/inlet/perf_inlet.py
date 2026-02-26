# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_inlet_drag import PerformancesInletDrag
from .perf_air_dynamic_pressure import PerformancesAirDynamicPressure
from .perf_max_ramp_pressure_efficiency import PerformancesMaxRamPressureEfficiency


class PerformancesInlet(om.Group):
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
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]

        self.add_subsystem(
            "inlet_drag",
            PerformancesInletDrag(
                pemfc_stack_bop_id=pemfc_stack_bop_id, number_of_points=number_of_points
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "air_dynamic_pressure",
            PerformancesAirDynamicPressure(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "max_ramp_pressure_efficiency",
            PerformancesMaxRamPressureEfficiency(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
