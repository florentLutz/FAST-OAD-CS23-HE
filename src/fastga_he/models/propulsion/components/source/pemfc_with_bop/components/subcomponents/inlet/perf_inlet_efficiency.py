# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesInletEfficiency(om.ExplicitComponent):
    """
    Computation of the inlet efficiency of pressure recovery.
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
            desc="Identifier of the air inlet",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input(
            "pressure_efficiency_difference",
            val=np.nan,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_ram_pressure_efficiency",
            val=np.nan,
            units="unitless",
        )

        self.add_output(
            "inlet_efficiency",
            val=0.85,
            units="unitless",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        air_inlet_id = self.options["air_inlet_id"]

        self.declare_partials(
            of="*",
            wrt="pressure_efficiency_difference",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=1.0,
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + self.options["pemfc_stack_bop_id"]
            + ":"
            + air_inlet_id
            + ":max_ram_pressure_efficiency",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        outputs["inlet_efficiency"] = (
            inputs["pressure_efficiency_difference"]
            + inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + air_inlet_id
                + ":max_ram_pressure_efficiency"
            ]
        )
