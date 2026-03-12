# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserExitAirSpeed(om.ExplicitComponent):
    """
    Computation of the average air speed in the diffuser.
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
            "throat_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio",
            val=np.nan,
            units="unitless",
        )

        self.add_output("exit_air_speed", val=0.3, units="m/s")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="throat_air_speed",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]

        outputs["exit_air_speed"] = (
            inputs["throat_air_speed"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + diffuser_id
                + ":area_ratio"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        diffuser_id = self.options["diffuser_id"]
        number_of_points = self.options["number_of_points"]

        partials["exit_air_speed", "throat_air_speed"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio"
        ] * np.ones(number_of_points)

        partials[
            "exit_air_speed",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + diffuser_id
            + ":area_ratio",
        ] = inputs["throat_air_speed"]
