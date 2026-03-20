# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirPressureDrop(om.ExplicitComponent):
    """
    Computation of the pressure drop at the air side of the heat exchanger.
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
            name="heat_exchanger_id",
            default=None,
            desc="Identifier of the heat exchanger",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="entrance_pressure_drop_coefficient",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="exit_pressure_drop_coefficient",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":free_flow_frontal_area_ratio",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="air_mass_velocity",
            units="kg/s/m**2",
            val=np.nan,
        )
        self.add_input(
            name="air_fanning_friction_factor",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="mean_air_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="air_inlet_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="air_outlet_density",
            units="kg/m**3",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            units="Pa",
            val=1e4,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]
        number_of_points = self.options["number_of_points"]

        k_entrance = inputs["entrance_pressure_drop_coefficient"]
        k_exit = inputs["exit_pressure_drop_coefficient"]
        sigma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":free_flow_frontal_area_ratio"
        ]
        air_mass_velocity = inputs["air_mass_velocity"]
        air_fanning_factor = inputs["air_fanning_friction_factor"]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
        ]
        air_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length"
        ]
        rho_air = inputs["mean_air_density"]
        rho_air_inlet = inputs["air_inlet_density"]
        rho_air_outlet = inputs["air_outlet_density"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop"
        ] = (
            (0.5 * air_mass_velocity**2.0)
            * (
                (-1.0 - sigma**2.0 + k_entrance) / rho_air_inlet
                + 4.0 * air_fanning_factor * air_flow_length / (rho_air * fin_hydraulic_diameter)
                + (1.0 + sigma**2.0 + k_exit) / rho_air_outlet
            )
            * np.ones(number_of_points)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]
        number_of_points = self.options["number_of_points"]

        k_entrance = inputs["entrance_pressure_drop_coefficient"]
        k_exit = inputs["exit_pressure_drop_coefficient"]
        sigma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":free_flow_frontal_area_ratio"
        ]
        air_mass_velocity = inputs["air_mass_velocity"]
        air_fanning_factor = inputs["air_fanning_friction_factor"]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
        ]
        air_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length"
        ]
        rho_air = inputs["mean_air_density"]
        rho_air_inlet = inputs["air_inlet_density"]
        rho_air_outlet = inputs["air_outlet_density"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "entrance_pressure_drop_coefficient",
        ] = 0.5 * air_mass_velocity**2.0 / rho_air_inlet * np.ones(number_of_points)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "exit_pressure_drop_coefficient",
        ] = 0.5 * air_mass_velocity**2.0 / rho_air_outlet * np.ones(number_of_points)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":free_flow_frontal_area_ratio",
        ] = (
            air_mass_velocity**2.0
            * sigma
            * (1.0 / rho_air_outlet - 1.0 / rho_air_inlet)
            * np.ones(number_of_points)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "air_mass_velocity",
        ] = (
            air_mass_velocity
            * (
                (1.0 - sigma**2.0 + k_entrance) / rho_air_inlet
                + 2.0 * (1.0 / rho_air_outlet - 1.0 / rho_air_inlet)
                + 4 * air_fanning_factor * air_flow_length / (rho_air * fin_hydraulic_diameter)
                - (1.0 - sigma**2.0 - k_exit) / rho_air_outlet
            )
            * np.ones(number_of_points)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "air_fanning_friction_factor",
        ] = (
            2.0
            * air_mass_velocity**2.0
            * air_flow_length
            / (rho_air * fin_hydraulic_diameter)
            * np.ones(number_of_points)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
        ] = (
            -2.0
            * air_mass_velocity**2.0
            * air_fanning_factor
            * air_flow_length
            / (rho_air * fin_hydraulic_diameter**2.0)
        ) * np.ones(number_of_points)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_flow_length",
        ] = (
            2.0
            * air_mass_velocity**2.0
            * air_fanning_factor
            / (rho_air * fin_hydraulic_diameter)
            * np.ones(number_of_points)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "mean_air_density",
        ] = (
            -2.0
            * air_mass_velocity**2.0
            * air_fanning_factor
            * air_flow_length
            / (rho_air**2.0 * fin_hydraulic_diameter)
        ) * np.ones(number_of_points)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "air_inlet_density",
        ] = (
            0.5
            * air_mass_velocity**2.0
            * (1.0 + sigma**2.0 - k_entrance)
            / rho_air_inlet**2.0
            * np.ones(number_of_points)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            "air_outlet_density",
        ] = (
            -0.5
            * air_mass_velocity**2.0
            * (1.0 + sigma**2.0 + k_exit)
            / rho_air_outlet**2.0
            * np.ones(number_of_points)
        )
