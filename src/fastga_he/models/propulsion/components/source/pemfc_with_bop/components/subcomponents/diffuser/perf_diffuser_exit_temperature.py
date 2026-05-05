# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserExitTemperature(om.ExplicitComponent):
    """
    Computation of the exit temperature of the diffuser.
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
            name="diffuser_id",
            default=None,
            desc="Identifier of the diffuser",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "throat_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "exit_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "exterior_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_air_specific_heat_capacity",
            val=np.nan,
            units="J/kg/K",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_air_density",
            val=np.nan,
            units="kg/m**3",
            shape=number_of_points,
        )

        self.add_output(
            "diffuser_exit_temperature",
            val=300.0,
            units="K",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        throat_air_speed = inputs["throat_air_speed"]
        exit_air_speed = inputs["exit_air_speed"]
        exterior_temperature = inputs["exterior_temperature"]
        diffuser_air_specific_heat_capacity = inputs["diffuser_air_specific_heat_capacity"]
        diffuser_air_density = inputs["diffuser_air_density"]
        pressure_drop = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop"
        ]

        outputs["diffuser_exit_temperature"] = (
            exterior_temperature
            + (
                0.5 * (throat_air_speed**2.0 - exit_air_speed**2.0)
                - pressure_drop / diffuser_air_density
            )
            / diffuser_air_specific_heat_capacity
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        throat_air_speed = inputs["throat_air_speed"]
        exit_air_speed = inputs["exit_air_speed"]
        diffuser_air_specific_heat_capacity = inputs["diffuser_air_specific_heat_capacity"]
        diffuser_air_density = inputs["diffuser_air_density"]
        pressure_drop = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop"
        ]

        partials["diffuser_exit_temperature", "throat_air_speed"] = (
            throat_air_speed / diffuser_air_specific_heat_capacity
        )

        partials["diffuser_exit_temperature", "exit_air_speed"] = (
            -exit_air_speed / diffuser_air_specific_heat_capacity
        )

        partials["diffuser_exit_temperature", "exterior_temperature"] = 1.0

        partials["diffuser_exit_temperature", "diffuser_air_specific_heat_capacity"] = (
            -(
                0.5 * (throat_air_speed**2.0 - exit_air_speed**2.0)
                - pressure_drop / diffuser_air_density
            )
            / diffuser_air_specific_heat_capacity**2.0
        )

        partials["diffuser_exit_temperature", "diffuser_air_density"] = pressure_drop / (
            diffuser_air_specific_heat_capacity * diffuser_air_density**2.0
        )

        partials[
            "diffuser_exit_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop",
        ] = -1.0 / (diffuser_air_specific_heat_capacity * diffuser_air_density)
