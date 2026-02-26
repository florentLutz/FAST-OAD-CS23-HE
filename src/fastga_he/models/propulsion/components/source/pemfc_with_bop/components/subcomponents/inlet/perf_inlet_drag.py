# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
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
from .perf_drag_coefficient_zero import PerformancesCDZeroInletMassFlow

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
        self.add_subsystem(
            "cd_zero_inlet_mass_flow",
            PerformancesCDZeroInletMassFlow(pemfc_stack_bop_id=pemfc_stack_bop_id),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inlet_drag",
            _PerformancesInletDrag(
                pemfc_stack_bop_id=pemfc_stack_bop_id, number_of_points=number_of_points
            ),
            promotes=["data:*"],
        )

        self.connect(
            "air_dynamic_viscosity.dynamic_viscosity",
            "max_boundary_layer_thickness.dynamic_viscosity",
        )
        self.connect(
            "air_mass_flow_ratio.air_mass_flow_ratio", "drag_correlation_factor.air_mass_flow_ratio"
        )
        self.connect(
            "modified_mass_flow_ratio.modified_mass_flow_ratio",
            "air_mass_flow_ratio.modified_mass_flow_ratio",
        )
        self.connect(
            "momentum_flow_correction_factor.momentum_flow_correction_factor",
            "inlet_drag.momentum_flow_correction_factor",
        )
        self.connect("air_mass_flow_ratio.air_mass_flow_ratio", "inlet_drag.air_mass_flow_ratio")
        self.connect(
            "drag_correlation_factor.drag_correlation_factor", "inlet_drag.drag_correlation_factor"
        )
        self.connect("drag_ksp_factor.k_sp_factor", "inlet_drag.k_sp_factor")
        self.connect("ramp_angle_factor.ramp_angle_factor", "inlet_drag.ramp_angle_factor")
        self.connect("mach_factor.mach_factor", "inlet_drag.mach_factor")
        self.connect(
            "cd_zero_inlet_mass_flow.cd_zero_inlet_mass_flow",
            "inlet_drag.cd_zero_inlet_mass_flow",
        )


class _PerformancesInletDrag(om.ExplicitComponent):
    """
    Computes the drag cause by the Inlet.
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

        self.add_input(
            "mach_factor",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "air_mass_flow_ratio",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
            val=np.nan,
            units="kg/s",
        )
        self.add_input(
            "drag_correlation_factor",
            val=1e-4,
            units="unitless",
        )
        self.add_input(
            "k_sp_factor",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "ramp_angle_factor",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "momentum_flow_correction_factor",
            val=np.nan,
            units="unitless",
        )
        self.add_input(
            "cd_zero_inlet_mass_flow",
            val=np.nan,
            units="unitless",
        )
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)

        self.add_output(
            "inlet_drag",
            val=500.0,
            units="N",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="inlet_drag",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="inlet_drag",
            wrt="true_airspeed",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        true_air_speed = inputs["true_airspeed"]
        mach_factor = inputs["mach_factor"]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]
        drag_correlation_factor = inputs["drag_correlation_factor"]
        k_sp_factor = inputs["k_sp_factor"]
        ramp_angle_factor = inputs["ramp_angle_factor"]
        momentum_flow_correction_factor = inputs["momentum_flow_correction_factor"]
        cd_zero_inlet_mass_flow = inputs["cd_zero_inlet_mass_flow"]
        air_consumption_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]

        outputs["inlet_drag"] = (
            0.5
            * (
                2.0 * momentum_flow_correction_factor * air_mass_flow_ratio
                + k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * air_consumption_max
            / air_mass_flow_ratio
            * true_air_speed
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]

        true_air_speed = inputs["true_airspeed"]
        mach_factor = inputs["mach_factor"]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]
        drag_correlation_factor = inputs["drag_correlation_factor"]
        k_sp_factor = inputs["k_sp_factor"]
        ramp_angle_factor = inputs["ramp_angle_factor"]
        momentum_flow_correction_factor = inputs["momentum_flow_correction_factor"]
        cd_zero_inlet_mass_flow = inputs["cd_zero_inlet_mass_flow"]
        air_consumption_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max"
        ]

        partials["inlet_drag", "true_airspeed"] = (
            0.5
            * (
                2.0 * momentum_flow_correction_factor * air_mass_flow_ratio
                + k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * air_consumption_max
            * np.ones(number_of_points)
            / air_mass_flow_ratio
        )

        partials["inlet_drag", "mach_factor"] = (
            0.5
            * k_sp_factor
            * ramp_angle_factor
            * cd_zero_inlet_mass_flow
            * air_consumption_max
            * true_air_speed
            / air_mass_flow_ratio
        )

        partials["inlet_drag", "air_mass_flow_ratio"] = (
            -0.5
            * (
                k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * air_consumption_max
            * true_air_speed
            / air_mass_flow_ratio**2.0
        )

        partials["inlet_drag", "drag_correlation_factor"] = (
            0.5 * air_consumption_max / air_mass_flow_ratio * true_air_speed
        )

        partials["inlet_drag", "k_sp_factor"] = (
            0.5
            * ramp_angle_factor
            * mach_factor
            * cd_zero_inlet_mass_flow
            * air_consumption_max
            * true_air_speed
            / air_mass_flow_ratio
        )

        partials["inlet_drag", "ramp_angle_factor"] = (
            0.5
            * k_sp_factor
            * mach_factor
            * cd_zero_inlet_mass_flow
            * air_consumption_max
            * true_air_speed
            / air_mass_flow_ratio
        )

        partials["inlet_drag", "momentum_flow_correction_factor"] = (
            air_consumption_max * true_air_speed
        )

        partials["inlet_drag", "cd_zero_inlet_mass_flow"] = (
            0.5
            * k_sp_factor
            * ramp_angle_factor
            * mach_factor
            * air_consumption_max
            * true_air_speed
            / air_mass_flow_ratio
        )

        partials[
            "inlet_drag",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_consumption_max",
        ] = (
            0.5
            * (
                2.0 * momentum_flow_correction_factor * air_mass_flow_ratio
                + k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * true_air_speed
            / air_mass_flow_ratio
        )
