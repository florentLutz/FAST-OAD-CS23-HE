# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .perf_max_inlet_boundary_layer_thickness import PerformancesMaxBoundaryLayerThickness
from .perf_throat_height_momentum_layer_thickness_ratio import (
    PerformancesThroatHeightMomentumBoundaryLayerThicknessRatio,
)
from .perf_boundary_layer_thickness_highlight_height_ratio import (
    PerformancesBoundaryLayerThicknessHighlightHeightRatio,
)
from .perf_momentum_flow_correction_factor import PerformancesMomentumFlowCorrectionFactor
from .perf_modified_mass_flow_ratio import PerformancesModifiedMassFlowRatio
from .perf_air_mass_flow_ratio import PerformancesAirMassFlowRatio
from .perf_drag_correlation_factor import PerformancesDragCorrelationFactor
from .perf_drag_ksp_factor import PerformancesDragKspFactor
from .perf_ramp_angle_factor import PerformancesRampAngleFactor
from .perf_mach_factor import PerformancesMachFactor

from ......loads.sm_pmsm.components.perf_air_dynamic_viscosity import (
    PerformancesAirDynamicViscosity,
)


class PerformancesInletDrag(om.Group):
    """
    Air inlet Drag computations.
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
            "air_dynamic_viscosity",
            PerformancesAirDynamicViscosity(number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "max_boundary_layer_thickness",
            PerformancesMaxBoundaryLayerThickness(
                pemfc_stack_bop_id=pemfc_stack_bop_id, number_of_points=number_of_points
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "throat_height_momentum_layer_thickness_ratio",
            PerformancesThroatHeightMomentumBoundaryLayerThicknessRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "boundary_layer_thickness_highlight_height_ratio",
            PerformancesBoundaryLayerThicknessHighlightHeightRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "momentum_flow_correction_factor",
            PerformancesMomentumFlowCorrectionFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "modified_mass_flow_ratio",
            PerformancesModifiedMassFlowRatio(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "air_mass_flow_ratio",
            PerformancesAirMassFlowRatio(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "drag_correlation_factor",
            PerformancesDragCorrelationFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "drag_ksp_factor",
            PerformancesDragKspFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "ramp_angle_factor",
            PerformancesRampAngleFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "mach_factor",
            PerformancesMachFactor(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )

        self.connect(
            "air_mass_flow_ratio.air_mass_flow_ratio", "drag_correlation_factor.air_mass_flow_ratio"
        )
