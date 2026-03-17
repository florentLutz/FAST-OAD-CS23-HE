# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserReynoldsNumber(om.ExplicitComponent):
    """
    Computation of the reynolds number in the diffuser.
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
            "average_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_air_density",
            val=np.nan,
            units="kg/m**3",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_air_dynamic_viscosity",
            val=np.nan,
            units="Pa*s",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter",
            val=np.nan,
            units="m",
        )

        self.add_output("air_reynolds_number", val=3e5, units="unitless", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        self.declare_partials(
            of="air_reynolds_number",
            wrt=["average_air_speed", "diffuser_air_density", "diffuser_air_dynamic_viscosity"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="air_reynolds_number",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        air_density = inputs["diffuser_air_density"]
        air_dynamic_viscosity = inputs["diffuser_air_dynamic_viscosity"]
        entry_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter"
        ]
        average_air_speed = inputs["average_air_speed"]

        outputs["air_reynolds_number"] = (
            air_density * average_air_speed * entry_hydraulic_diameter / air_dynamic_viscosity
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        air_density = inputs["diffuser_air_density"]
        air_dynamic_viscosity = inputs["diffuser_air_dynamic_viscosity"]
        entry_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter"
        ]
        average_air_speed = inputs["average_air_speed"]

        partials["air_reynolds_number", "average_air_speed"] = (
            air_density * entry_hydraulic_diameter / air_dynamic_viscosity
        )

        partials["air_reynolds_number", "diffuser_air_density"] = (
            average_air_speed * entry_hydraulic_diameter / air_dynamic_viscosity
        )

        partials["air_reynolds_number", "diffuser_air_dynamic_viscosity"] = (
            -air_density * average_air_speed * entry_hydraulic_diameter / air_dynamic_viscosity**2.0
        )

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":entry_hydraulic_diameter",
        ] = air_density * average_air_speed / air_dynamic_viscosity
