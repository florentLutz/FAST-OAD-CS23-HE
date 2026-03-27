# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzlePressureDrop(om.ExplicitComponent):
    """
    Computation of the pressure drop of the nozzle.
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
            name="nozzle_id",
            default=None,
            desc="Identifier of the nozzle",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_input(
            "nozzle_friction_loss_coefficient",
            val=np.nan,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "nozzle_contraction_loss_coefficient",
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
            "nozzle_air_density",
            val=np.nan,
            units="kg/m**3",
            shape=number_of_points,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop",
            val=2000.0,
            units="Pa",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt=["average_air_speed", "nozzle_air_density", "nozzle_friction_loss_coefficient"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="nozzle_contraction_loss_coefficient",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        nozzle_friction_loss_coefficient = inputs["nozzle_friction_loss_coefficient"]
        nozzle_contraction_loss_coefficient = inputs["nozzle_contraction_loss_coefficient"]
        average_air_speed = inputs["average_air_speed"]
        nozzle_air_density = inputs["nozzle_air_density"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop"
        ] = (
            0.5
            * nozzle_air_density
            * average_air_speed**2.0
            * (nozzle_friction_loss_coefficient + nozzle_contraction_loss_coefficient)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        nozzle_friction_loss_coefficient = inputs["nozzle_friction_loss_coefficient"]
        nozzle_contraction_loss_coefficient = inputs["nozzle_contraction_loss_coefficient"]
        average_air_speed = inputs["average_air_speed"]
        nozzle_air_density = inputs["nozzle_air_density"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop",
            "nozzle_friction_loss_coefficient",
        ] = 0.5 * nozzle_air_density * average_air_speed**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop",
            "nozzle_contraction_loss_coefficient",
        ] = 0.5 * nozzle_air_density * average_air_speed**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop",
            "average_air_speed",
        ] = (
            nozzle_air_density
            * average_air_speed
            * (nozzle_friction_loss_coefficient + nozzle_contraction_loss_coefficient)
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop",
            "nozzle_air_density",
        ] = (
            0.5
            * average_air_speed**2.0
            * (nozzle_friction_loss_coefficient + nozzle_contraction_loss_coefficient)
        )
