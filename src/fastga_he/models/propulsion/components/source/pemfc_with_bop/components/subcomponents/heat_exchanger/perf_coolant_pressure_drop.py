# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

MINOR_LOSS_FACTOR = 0.375


class PerformancesCoolantPressureDrop(om.ExplicitComponent):
    """
    Computation of the pressure drop at the coolant side of the heat exchanger.
    """

    def initialize(self):
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
            name="coolant_mass_velocity",
            units="kg/s/m**2",
            val=np.nan,
        )
        self.add_input(
            name="coolant_fanning_friction_factor",
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
            + ":coolant_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="mean_coolant_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="coolant_inlet_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="coolant_outlet_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity",
            units="m/s",
            val=1.5,
            desc="Pipe flow velocity",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            units="Pa",
            val=200.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        k_entrance = inputs["entrance_pressure_drop_coefficient"]
        k_exit = inputs["exit_pressure_drop_coefficient"]
        sigma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":free_flow_frontal_area_ratio"
        ]
        coolant_mass_velocity = inputs["coolant_mass_velocity"]
        coolant_fanning_factor = inputs["coolant_fanning_friction_factor"]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
        ]
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]
        rho_coolant = inputs["mean_coolant_density"]
        rho_coolant_inlet = inputs["coolant_inlet_density"]
        rho_coolant_outlet = inputs["coolant_outlet_density"]
        flow_velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop"
        ] = (0.5 * coolant_mass_velocity**2.0) * (
            (-1.0 - sigma**2.0 + k_entrance) / rho_coolant_inlet
            + 4.0
            * coolant_fanning_factor
            * coolant_flow_length
            / (rho_coolant * fin_hydraulic_diameter)
            + (1.0 + sigma**2.0 + k_exit) / rho_coolant_outlet
        ) + rho_coolant * flow_velocity**2.0 * MINOR_LOSS_FACTOR

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        k_entrance = inputs["entrance_pressure_drop_coefficient"]
        k_exit = inputs["exit_pressure_drop_coefficient"]
        sigma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":free_flow_frontal_area_ratio"
        ]
        coolant_mass_velocity = inputs["coolant_mass_velocity"]
        coolant_fanning_factor = inputs["coolant_fanning_friction_factor"]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
        ]
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length"
        ]
        rho_coolant = inputs["mean_coolant_density"]
        rho_coolant_inlet = inputs["coolant_inlet_density"]
        rho_coolant_outlet = inputs["coolant_outlet_density"]
        flow_velocity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "entrance_pressure_drop_coefficient",
        ] = 0.5 * coolant_mass_velocity**2.0 / rho_coolant_inlet

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "exit_pressure_drop_coefficient",
        ] = 0.5 * coolant_mass_velocity**2.0 / rho_coolant_outlet

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":free_flow_frontal_area_ratio",
        ] = (
            coolant_mass_velocity**2.0
            * sigma
            * (1.0 / rho_coolant_outlet - 1.0 / rho_coolant_inlet)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "coolant_mass_velocity",
        ] = coolant_mass_velocity * (
            (1.0 - sigma**2.0 + k_entrance) / rho_coolant_inlet
            + 2.0 * (1.0 / rho_coolant_outlet - 1.0 / rho_coolant_inlet)
            + 4
            * coolant_fanning_factor
            * coolant_flow_length
            / (rho_coolant * fin_hydraulic_diameter)
            - (1.0 - sigma**2.0 - k_exit) / rho_coolant_outlet
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "coolant_fanning_friction_factor",
        ] = (
            2.0
            * coolant_mass_velocity**2.0
            * coolant_flow_length
            / (rho_coolant * fin_hydraulic_diameter)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
        ] = (
            -2.0
            * coolant_mass_velocity**2.0
            * coolant_fanning_factor
            * coolant_flow_length
            / (rho_coolant * fin_hydraulic_diameter**2.0)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_flow_length",
        ] = (
            2.0
            * coolant_mass_velocity**2.0
            * coolant_fanning_factor
            / (rho_coolant * fin_hydraulic_diameter)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "mean_coolant_density",
        ] = (
            -2.0
            * coolant_mass_velocity**2.0
            * coolant_fanning_factor
            * coolant_flow_length
            / (rho_coolant**2.0 * fin_hydraulic_diameter)
        ) + flow_velocity**2.0 * MINOR_LOSS_FACTOR

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "coolant_inlet_density",
        ] = (
            0.5
            * coolant_mass_velocity**2.0
            * (1.0 + sigma**2.0 - k_entrance)
            / rho_coolant_inlet**2.0
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "coolant_outlet_density",
        ] = (
            -0.5
            * coolant_mass_velocity**2.0
            * (1.0 + sigma**2.0 + k_exit)
            / rho_coolant_outlet**2.0
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":coolant_pressure_drop",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:transport_velocity",
        ] = 2.0 * rho_coolant * flow_velocity * MINOR_LOSS_FACTOR
