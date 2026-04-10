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
            "max_boundary_layer_thickness",
            PerformancesMaxBoundaryLayerThickness(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "throat_height_momentum_layer_thickness_ratio",
            PerformancesThroatHeightMomentumBoundaryLayerThicknessRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "boundary_layer_thickness_highlight_height_ratio",
            PerformancesBoundaryLayerThicknessHighlightHeightRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "momentum_flow_correction_factor",
            PerformancesMomentumFlowCorrectionFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "modified_mass_flow_ratio",
            PerformancesModifiedMassFlowRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "air_mass_flow_ratio",
            PerformancesAirMassFlowRatio(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "drag_correlation_factor",
            PerformancesDragCorrelationFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "drag_ksp_factor",
            PerformancesDragKspFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "ramp_angle_factor",
            PerformancesRampAngleFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "mach_factor",
            PerformancesMachFactor(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "cd_zero_inlet_mass_flow",
            PerformancesCDZeroInletMassFlow(
                pemfc_stack_bop_id=pemfc_stack_bop_id, air_inlet_id=air_inlet_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "inlet_drag",
            _PerformancesInletDrag(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                number_of_points=number_of_points,
                air_inlet_id=air_inlet_id,
            ),
            promotes=["*"],
        )

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 5
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()


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
            "inlet_air_mass_flow",
            val=np.nan,
            units="kg/s",
            shape=number_of_points,
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
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            val=0.0,
            units="N",
            shape=number_of_points,
            lower=0.0,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            wrt=["true_airspeed", "inlet_air_mass_flow"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        true_air_speed = inputs["true_airspeed"]
        mach_factor = inputs["mach_factor"]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]
        drag_correlation_factor = inputs["drag_correlation_factor"]
        k_sp_factor = inputs["k_sp_factor"]
        ramp_angle_factor = inputs["ramp_angle_factor"]
        momentum_flow_correction_factor = inputs["momentum_flow_correction_factor"]
        cd_zero_inlet_mass_flow = inputs["cd_zero_inlet_mass_flow"]
        air_mass_flow = inputs["inlet_air_mass_flow"]

        unclipped_drag = (
            0.5
            * (
                2.0 * momentum_flow_correction_factor * air_mass_flow_ratio
                + k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * air_mass_flow
            / air_mass_flow_ratio
            * true_air_speed
        )

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag"
        ] = np.clip(unclipped_drag, 0.0, 150)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        true_air_speed = inputs["true_airspeed"]
        mach_factor = inputs["mach_factor"]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]
        drag_correlation_factor = inputs["drag_correlation_factor"]
        k_sp_factor = inputs["k_sp_factor"]
        ramp_angle_factor = inputs["ramp_angle_factor"]
        momentum_flow_correction_factor = inputs["momentum_flow_correction_factor"]
        cd_zero_inlet_mass_flow = inputs["cd_zero_inlet_mass_flow"]
        air_mass_flow = inputs["inlet_air_mass_flow"]

        unclipped_drag = (
            0.5
            * (
                2.0 * momentum_flow_correction_factor * air_mass_flow_ratio
                + k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * air_mass_flow
            / air_mass_flow_ratio
            * true_air_speed
        )

        clipped_drag = np.clip(unclipped_drag, 0.0, 150.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "true_airspeed",
        ] = np.where(
            unclipped_drag == clipped_drag,
            0.5
            * (
                2.0 * momentum_flow_correction_factor * air_mass_flow_ratio
                + k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * air_mass_flow
            / air_mass_flow_ratio,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "mach_factor",
        ] = np.where(
            unclipped_drag == clipped_drag,
            0.5
            * k_sp_factor
            * ramp_angle_factor
            * cd_zero_inlet_mass_flow
            * air_mass_flow
            * true_air_speed
            / air_mass_flow_ratio,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "air_mass_flow_ratio",
        ] = np.where(
            unclipped_drag == clipped_drag,
            -0.5
            * (
                k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * air_mass_flow
            * true_air_speed
            / air_mass_flow_ratio**2.0,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "drag_correlation_factor",
        ] = np.where(
            unclipped_drag == clipped_drag,
            0.5 * air_mass_flow / air_mass_flow_ratio * true_air_speed,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "k_sp_factor",
        ] = np.where(
            unclipped_drag == clipped_drag,
            0.5
            * ramp_angle_factor
            * mach_factor
            * cd_zero_inlet_mass_flow
            * air_mass_flow
            * true_air_speed
            / air_mass_flow_ratio,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "ramp_angle_factor",
        ] = np.where(
            unclipped_drag == clipped_drag,
            0.5
            * k_sp_factor
            * mach_factor
            * cd_zero_inlet_mass_flow
            * air_mass_flow
            * true_air_speed
            / air_mass_flow_ratio,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "momentum_flow_correction_factor",
        ] = air_mass_flow * true_air_speed

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "cd_zero_inlet_mass_flow",
        ] = np.where(
            unclipped_drag == clipped_drag,
            0.5
            * k_sp_factor
            * ramp_angle_factor
            * mach_factor
            * air_mass_flow
            * true_air_speed
            / air_mass_flow_ratio,
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":drag",
            "inlet_air_mass_flow",
        ] = np.where(
            unclipped_drag == clipped_drag,
            0.5
            * (
                2.0 * momentum_flow_correction_factor * air_mass_flow_ratio
                + k_sp_factor * ramp_angle_factor * mach_factor * cd_zero_inlet_mass_flow
                + drag_correlation_factor
            )
            * true_air_speed
            / air_mass_flow_ratio,
            1e-6,
        )
