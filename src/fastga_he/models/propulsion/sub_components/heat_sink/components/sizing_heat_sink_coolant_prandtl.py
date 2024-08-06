# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om


class SizingHeatSinkCoolantPrandtl(om.ExplicitComponent):
    """
    Computation of the Prandtl number of the selected coolant. Fluid properties were taken at 20
    degC but thy change with temperature. The thorough TMS model should take that in account.
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
            name=prefix + ":heat_sink:coolant:specific_heat_capacity",
            units="J/degK/kg",
            val=3260.0,
            desc="Specific heat capacity of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:thermal_conductivity",
            units="W/m/degK",
            val=0.402,
            desc="Thermal conductivity of the coolant fluid",
        )
        self.add_input(
            name=prefix + ":heat_sink:coolant:dynamic_viscosity",
            units="Pa*s",
            val=4.87e-3,
            desc="Dynamic viscosity of the coolant fluid",
        )

        self.add_output(
            name=prefix + ":heat_sink:coolant:Prandtl_number",
            val=40.0,
            desc="Prandtl number of the coolant",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        specific_heat_capacity_coolant = inputs[
            prefix + ":heat_sink:coolant:specific_heat_capacity"
        ]
        dynamic_viscosity_coolant = inputs[prefix + ":heat_sink:coolant:dynamic_viscosity"]
        thermal_conductivity_coolant = inputs[prefix + ":heat_sink:coolant:thermal_conductivity"]

        outputs[prefix + ":heat_sink:coolant:Prandtl_number"] = (
            specific_heat_capacity_coolant
            * dynamic_viscosity_coolant
            / thermal_conductivity_coolant
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        specific_heat_capacity_coolant = inputs[
            prefix + ":heat_sink:coolant:specific_heat_capacity"
        ]
        dynamic_viscosity_coolant = inputs[prefix + ":heat_sink:coolant:dynamic_viscosity"]
        thermal_conductivity_coolant = inputs[prefix + ":heat_sink:coolant:thermal_conductivity"]

        partials[
            prefix + ":heat_sink:coolant:Prandtl_number",
            prefix + ":heat_sink:coolant:specific_heat_capacity",
        ] = dynamic_viscosity_coolant / thermal_conductivity_coolant
        partials[
            prefix + ":heat_sink:coolant:Prandtl_number",
            prefix + ":heat_sink:coolant:dynamic_viscosity",
        ] = specific_heat_capacity_coolant / thermal_conductivity_coolant
        partials[
            prefix + ":heat_sink:coolant:Prandtl_number",
            prefix + ":heat_sink:coolant:thermal_conductivity",
        ] = -(
            specific_heat_capacity_coolant
            * dynamic_viscosity_coolant
            / thermal_conductivity_coolant**2.0
        )
