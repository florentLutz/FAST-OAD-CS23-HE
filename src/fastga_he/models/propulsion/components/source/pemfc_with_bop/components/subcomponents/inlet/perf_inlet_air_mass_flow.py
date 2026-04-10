# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga_he.models.propulsion.components.source.pemfc_with_bop.components.subcomponents.fluid_characteristics import (
    FluidSpecificHeatCapacity,
)


class PerformancesAirInletAirMassFlow(om.ExplicitComponent):
    """
    Compute the total air mass flow rate at the air inlet.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air_inlet",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input(
            "air_consumption",
            units="kg/s",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
            val=10.0,
            units="unitless",
        )

        self.add_output("inlet_air_mass_flow", val=5.0, units="kg/s", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials("*", "*", method="exact")
        self.declare_partials(
            "inlet_air_mass_flow",
            "air_consumption",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        outputs["inlet_air_mass_flow"] = np.clip(
            inputs["air_consumption"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":mass_flow_factor"
            ],
            0.001,
            np.inf,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]
        number_of_points = self.options["number_of_points"]

        unclipped_air_mass_flow = (
            inputs["air_consumption"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":mass_flow_factor"
            ]
        )

        clipped_air_mass_flow = np.clip(unclipped_air_mass_flow, 0.001, np.inf)

        partials["inlet_air_mass_flow", "air_consumption"] = np.where(
            unclipped_air_mass_flow == clipped_air_mass_flow,
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":mass_flow_factor"
            ]
            * np.ones(number_of_points),
            1e-6,
        )

        partials[
            "inlet_air_mass_flow",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":mass_flow_factor",
        ] = np.where(
            unclipped_air_mass_flow == clipped_air_mass_flow, inputs["air_consumption"], 1e-6
        )
