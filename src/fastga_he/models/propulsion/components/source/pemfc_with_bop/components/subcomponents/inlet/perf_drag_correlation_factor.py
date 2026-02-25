# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import jax
from scipy.optimize import fsolve


class PerformancesDragCorrelationFactor(om.ExplicitComponent):
    """
    Computation of the drag correlation factor.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "air_mass_flow_ratio",
            val=1e-4,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "drag_correlation_factor",
            val=1e-4,
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach"
        ]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]

        # Solve the implicit equation for the drag correlation factor
        corr_drag_initial_guess = 0.1  # Initial guess for the drag correlation factor
        self._cached_drag_correlation_factor = fsolve(
            lambda corr_drag: drag_correlation_equation_to_solve(
                corr_drag, design_mach, air_mass_flow_ratio
            ),
            corr_drag_initial_guess,
        )[0]

        outputs["drag_correlation_factor"] = self._cached_drag_correlation_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach"
        ]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]
        drag_correlation_factor = self._cached_drag_correlation_factor

        # Partial derivatives
        df_dcorr_drag = jax.grad(drag_correlation_equation_to_solve, argnums=0)  # ∂f/∂corr_drag
        df_dmach = jax.grad(drag_correlation_equation_to_solve, argnums=1)  # ∂f/∂design_mach
        df_dairflowratio = jax.grad(drag_correlation_equation_to_solve, argnums=2)
        # ∂f/∂air_mass_flow_ratio

        # Evaluate derivatives at solution point
        denom = float(df_dcorr_drag(drag_correlation_factor, design_mach, air_mass_flow_ratio))
        numer_mach = float(df_dmach(drag_correlation_factor, design_mach, air_mass_flow_ratio))
        numer_airflowratio = float(
            df_dairflowratio(drag_correlation_factor, design_mach, air_mass_flow_ratio)
        )

        # Implicit differentiation
        partials[
            "drag_correlation_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":air_inlet:design_mach",
        ] = -numer_mach / denom

        partials["drag_correlation_factor", "air_mass_flow_ratio"] = -numer_airflowratio / denom


def drag_correlation_equation_to_solve(corr_drag, design_mach, air_mass_flow_ratio):
    """Implicit equation: f(corr_drag, design_mach, air_mass_flow_ratio) = 0"""
    return (
        0.99575
        + 0.72927 * design_mach**2.0
        + 34.61116 * corr_drag
        - 36.33161 * corr_drag**2.0
        + 154.13563 * corr_drag**3.0 * design_mach
        + 2.35051 * design_mach**4.0
        - 3.67345 * design_mach**3.0
        - 53.10867 * corr_drag * design_mach
        + 24.61205 * corr_drag * design_mach**3.0
        - air_mass_flow_ratio
    )
