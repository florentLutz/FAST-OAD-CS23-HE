# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDragCorrelationFactor(om.ImplicitComponent):
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
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input(
            "air_mass_flow_ratio",
            val=1e-4,
            units="unitless",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "drag_correlation_factor",
            units="unitless",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach"
        ]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]
        corr_drag = outputs["drag_correlation_factor"]

        residuals["drag_correlation_factor"] = (
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

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        design_mach = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach"
        ]
        air_mass_flow_ratio = inputs["air_mass_flow_ratio"]
        corr_drag = outputs["drag_correlation_factor"]

        jacobian["drag_correlation_factor", "drag_correlation_factor"] = (
            34.61116
            - 72.66322 * corr_drag
            + 462.40689 * corr_drag**2.0 * design_mach
            - 53.10867 * design_mach
            + 24.61205 * design_mach**3.0
        )

        jacobian["drag_correlation_factor", "air_mass_flow_ratio"] = -1.0

        jacobian[
            "drag_correlation_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_mach",
        ] = (
            1.45854 * design_mach
            + 154.13563 * corr_drag**3.0
            + 9.40204 * design_mach**3.0
            - 11.02035 * design_mach**2.0
            - 53.10867 * corr_drag
            + 73.83615 * corr_drag * design_mach**2.0
        )
