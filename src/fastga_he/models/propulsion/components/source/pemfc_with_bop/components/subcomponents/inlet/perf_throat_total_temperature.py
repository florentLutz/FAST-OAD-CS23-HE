# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesThroatTemperature(om.ExplicitComponent):
    """
    Computation of the ambient throat temperature.
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

        self.add_input("mach", val=np.nan, shape=number_of_points)
        self.add_input(
            "exterior_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":specific_heat_ratio",
            val=1.4,
            units="unitless",
        )

        self.add_output(
            "throat_total_temperature",
            val=300.0,
            units="K",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        self.declare_partials(
            of="*",
            wrt=["mach", "exterior_temperature"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":specific_heat_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        exterior_temperature = inputs["exterior_temperature"]
        mach = inputs["mach"]
        gamma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":specific_heat_ratio"
        ]

        outputs["throat_total_temperature"] = exterior_temperature * (
            1.0 + (gamma - 1.0) * 0.5 * mach**2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        exterior_temperature = inputs["exterior_temperature"]
        mach = inputs["mach"]
        gamma = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":specific_heat_ratio"
        ]

        partials["throat_total_temperature", "exterior_temperature"] = (
            1.0 + (gamma - 1.0) * 0.5 * mach**2.0
        )

        partials["throat_total_temperature", "mach"] = exterior_temperature * (gamma - 1.0) * mach

        partials[
            "throat_total_temperature",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":specific_heat_ratio",
        ] = exterior_temperature * 0.5 * mach**2.0
