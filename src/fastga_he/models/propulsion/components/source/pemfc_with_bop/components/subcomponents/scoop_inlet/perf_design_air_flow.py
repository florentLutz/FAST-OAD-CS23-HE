# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesInletDesignAirFlow(om.ExplicitComponent):
    """
    Computation of the design air flow of the inlet.
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
            "total_air_mass_flow",
            val=np.nan,
            units="kg/s",
            shape=number_of_points,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_air_mass_flow",
            val=2.1,
            units="kg/s",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_air_mass_flow"
        ] = np.max(inputs["total_air_mass_flow"])

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        design_air_mass_flow = np.max(inputs["total_air_mass_flow"])

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":design_air_mass_flow",
            "total_air_mass_flow",
        ] = np.where(design_air_mass_flow == inputs["total_air_mass_flow"], 1.0, 1e-6)
