# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHeatSinkTubeInnerDiameter(om.ExplicitComponent):
    """
    Computation of the inner diameter of the tube running in the heat sink based on the cooling
    capabilities necessary. Method from :cite:`giraud:2014`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a heatsink",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":heat_sink:coolant:max_mass_flow",
            val=np.nan,
            units="m**3/s",
            desc="Maximum mass flow necessary to cool the inverter",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:Prandtl_number",
            val=np.nan,
            desc="Prandtl number of the coolant",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:dynamic_viscosity",
            units="Pa*s",
            val=4.87e-3,
            desc="Dynamic viscosity of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:density",
            units="kg/m**3",
            val=1082.0,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:thermal_conductivity",
            units="W/m/degK",
            val=0.402,
            desc="Thermal conductivity of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:temperature_in_rating",
            units="degK",
            val=np.nan,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:temperature_out_rating",
            units="degK",
            val=np.nan,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:temperature_rating",
            units="degK",
            val=np.nan,
            desc="Density of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:tube:length",
            units="m",
            val=np.nan,
            desc="Length of the tube which is useful for the cooling of the inverter",
        )
        self.add_input(
            name=prefix + ":dissipable_heat",
            units="W",
            val=np.nan,
            desc="Maximum power losses of the inverter (all modules)",
        )

        self.add_output(
            name=prefix + ":heat_sink:tube:inner_diameter",
            units="m",
            val=0.1,
            desc="Inner diameter of the tube for the cooling of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        inverter_losses_max = inputs[prefix + ":dissipable_heat"]
        density_coolant = inputs[prefix + ":heat_sink:coolant:density"]
        t_max_out_coolant = inputs[prefix + ":heat_sink:coolant:temperature_out_rating"]
        t_max_in_coolant = inputs[prefix + ":heat_sink:coolant:temperature_in_rating"]
        t_max_heat_sink = inputs[prefix + ":heat_sink:temperature_rating"]
        mass_flow_max_coolant = inputs[prefix + ":heat_sink:coolant:max_mass_flow"]
        prandtl_coolant = inputs[prefix + ":heat_sink:coolant:Prandtl_number"]
        dynamic_viscosity_coolant = inputs[prefix + ":heat_sink:coolant:dynamic_viscosity"]
        thermal_conductivity_coolant = inputs[prefix + ":heat_sink:coolant:thermal_conductivity"]
        tube_length = inputs[prefix + ":heat_sink:tube:length"]

        # Spatial averages
        t_max_avg_coolant = (t_max_in_coolant + t_max_out_coolant) / 2.0
        y_term = inverter_losses_max / (tube_length * np.pi * (t_max_heat_sink - t_max_avg_coolant))

        inner_diameter = (
            (
                thermal_conductivity_coolant
                * 0.023
                * prandtl_coolant**0.4
                * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
            )
            / (y_term * (dynamic_viscosity_coolant * np.pi) ** 0.8)
        ) ** (1.0 / 0.8)

        outputs[prefix + ":heat_sink:tube:inner_diameter"] = inner_diameter

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        inverter_losses_max = inputs[prefix + ":dissipable_heat"]
        density_coolant = inputs[prefix + ":heat_sink:coolant:density"]
        t_max_out_coolant = inputs[prefix + ":heat_sink:coolant:temperature_out_rating"]
        t_max_in_coolant = inputs[prefix + ":heat_sink:coolant:temperature_in_rating"]
        t_max_heat_sink = inputs[prefix + ":heat_sink:temperature_rating"]
        mass_flow_max_coolant = inputs[prefix + ":heat_sink:coolant:max_mass_flow"]
        prandtl_coolant = inputs[prefix + ":heat_sink:coolant:Prandtl_number"]
        dynamic_viscosity_coolant = inputs[prefix + ":heat_sink:coolant:dynamic_viscosity"]
        thermal_conductivity_coolant = inputs[prefix + ":heat_sink:coolant:thermal_conductivity"]
        tube_length = inputs[prefix + ":heat_sink:tube:length"]

        t_max_avg_coolant = (t_max_in_coolant + t_max_out_coolant) / 2.0
        y_term = inverter_losses_max / (tube_length * np.pi * (t_max_heat_sink - t_max_avg_coolant))

        d_y_d_losses = 1.0 / (tube_length * np.pi * (t_max_heat_sink - t_max_avg_coolant))
        d_y_d_length = -inverter_losses_max / (
            tube_length**2.0 * np.pi * (t_max_heat_sink - t_max_avg_coolant)
        )
        d_y_d_t_max_hs = -inverter_losses_max / (
            tube_length * np.pi * (t_max_heat_sink - t_max_avg_coolant) ** 2.0
        )
        d_y_d_t_max_in = inverter_losses_max / (
            2.0 * tube_length * np.pi * (t_max_heat_sink - t_max_avg_coolant) ** 2.0
        )
        d_y_d_t_max_out = inverter_losses_max / (
            2.0 * tube_length * np.pi * (t_max_heat_sink - t_max_avg_coolant) ** 2.0
        )

        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:coolant:max_mass_flow",
        ] = (
            (
                thermal_conductivity_coolant
                * 0.023
                * prandtl_coolant**0.4
                * (4.0 * density_coolant) ** 0.8
            )
            / (y_term * (dynamic_viscosity_coolant * np.pi) ** 0.8)
        ) ** (1.0 / 0.8)
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:coolant:Prandtl_number",
        ] = (
            (
                (
                    thermal_conductivity_coolant
                    * 0.023
                    * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                )
                / (y_term * (dynamic_viscosity_coolant * np.pi) ** 0.8)
            )
            ** (1.0 / 0.8)
            * 0.4
            / 0.8
            * prandtl_coolant ** (0.4 / 0.8 - 1.0)
        )
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:coolant:dynamic_viscosity",
        ] = (
            -(
                (
                    (
                        thermal_conductivity_coolant
                        * 0.023
                        * prandtl_coolant**0.4
                        * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                    )
                    / (y_term * np.pi**0.8)
                )
                ** (1.0 / 0.8)
            )
            / dynamic_viscosity_coolant**2.0
        )
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:coolant:density",
        ] = (
            (
                thermal_conductivity_coolant
                * 0.023
                * prandtl_coolant**0.4
                * (4.0 * mass_flow_max_coolant) ** 0.8
            )
            / (y_term * (dynamic_viscosity_coolant * np.pi) ** 0.8)
        ) ** (1.0 / 0.8)
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:coolant:thermal_conductivity",
        ] = (
            (
                (
                    0.023
                    * prandtl_coolant**0.4
                    * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                )
                / (y_term * (dynamic_viscosity_coolant * np.pi) ** 0.8)
            )
            ** (1.0 / 0.8)
            * (1.0 / 0.8)
            * thermal_conductivity_coolant ** (1.0 / 0.8 - 1.0)
        )
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:coolant:temperature_in_rating",
        ] = -(
            (
                (
                    thermal_conductivity_coolant
                    * 0.023
                    * prandtl_coolant**0.4
                    * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                )
                / (dynamic_viscosity_coolant * np.pi) ** 0.8
            )
            ** (1.0 / 0.8)
            * y_term ** (-1.0 / 0.8 - 1.0)
            * 1.0
            / 0.8
            * d_y_d_t_max_in
        )
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:coolant:temperature_out_rating",
        ] = -(
            (
                (
                    thermal_conductivity_coolant
                    * 0.023
                    * prandtl_coolant**0.4
                    * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                )
                / (dynamic_viscosity_coolant * np.pi) ** 0.8
            )
            ** (1.0 / 0.8)
            * y_term ** (-1.0 / 0.8 - 1.0)
            * 1.0
            / 0.8
            * d_y_d_t_max_out
        )
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:temperature_rating",
        ] = -(
            (
                (
                    thermal_conductivity_coolant
                    * 0.023
                    * prandtl_coolant**0.4
                    * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                )
                / (dynamic_viscosity_coolant * np.pi) ** 0.8
            )
            ** (1.0 / 0.8)
            * y_term ** (-1.0 / 0.8 - 1.0)
            * 1.0
            / 0.8
            * d_y_d_t_max_hs
        )
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":heat_sink:tube:length",
        ] = -(
            (
                (
                    thermal_conductivity_coolant
                    * 0.023
                    * prandtl_coolant**0.4
                    * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                )
                / (dynamic_viscosity_coolant * np.pi) ** 0.8
            )
            ** (1.0 / 0.8)
            * y_term ** (-1.0 / 0.8 - 1.0)
            * 1.0
            / 0.8
            * d_y_d_length
        )
        partials[
            prefix + ":heat_sink:tube:inner_diameter",
            prefix + ":dissipable_heat",
        ] = -(
            (
                (
                    thermal_conductivity_coolant
                    * 0.023
                    * prandtl_coolant**0.4
                    * (4.0 * density_coolant * mass_flow_max_coolant) ** 0.8
                )
                / (dynamic_viscosity_coolant * np.pi) ** 0.8
            )
            ** (1.0 / 0.8)
            * y_term ** (-1.0 / 0.8 - 1.0)
            * 1.0
            / 0.8
            * d_y_d_losses
        )
