# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserPressureDrop(om.ExplicitComponent):
    """
    Computation of the pressure drop of the diffuser.
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
            "diffuser_friction_loss_coefficient",
            val=0.3,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_expansion_loss_coefficient",
            val=np.nan,
            units="unitless",
        )
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

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop",
            val=0.3,
            units="Pa",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt=["average_air_speed", "diffuser_air_density", "diffuser_friction_loss_coefficient"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="diffuser_expansion_loss_coefficient",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        diffuser_friction_loss_coefficient = inputs["diffuser_friction_loss_coefficient"]
        diffuser_expansion_loss_coefficient = inputs["diffuser_expansion_loss_coefficient"]
        average_air_speed = inputs["average_air_speed"]
        diffuser_air_density = inputs["diffuser_air_density"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop"
        ] = (
            0.5
            * diffuser_air_density
            * average_air_speed**2.0
            * (diffuser_friction_loss_coefficient + diffuser_expansion_loss_coefficient)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        diffuser_friction_loss_coefficient = inputs["diffuser_friction_loss_coefficient"]
        diffuser_expansion_loss_coefficient = inputs["diffuser_expansion_loss_coefficient"]
        average_air_speed = inputs["average_air_speed"]
        diffuser_air_density = inputs["diffuser_air_density"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop",
            "diffuser_friction_loss_coefficient",
        ] = 0.5 * diffuser_air_density * average_air_speed**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop",
            "diffuser_expansion_loss_coefficient",
        ] = 0.5 * diffuser_air_density * average_air_speed**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop",
            "average_air_speed",
        ] = (
            diffuser_air_density
            * average_air_speed
            * (diffuser_friction_loss_coefficient + diffuser_expansion_loss_coefficient)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":air_pressure_drop",
            "diffuser_air_density",
        ] = (
            0.5
            * average_air_speed**2.0
            * (diffuser_friction_loss_coefficient + diffuser_expansion_loss_coefficient)
        )
