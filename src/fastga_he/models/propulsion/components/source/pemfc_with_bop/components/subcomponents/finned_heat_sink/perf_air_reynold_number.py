# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirReynoldsNumber(om.ExplicitComponent):
    """
    Computation of the air Reynolds number.
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
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        self.add_input(name="true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_spacing",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_length",
            units="m",
            val=np.nan,
        )
        self.add_input("dynamic_viscosity", val=np.nan, units="Pa*s", shape=number_of_points)

        self.add_output(
            name="reynolds_number",
            units="unitless",
            val=2400.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        self.declare_partials("*", "*", method="exact")
        self.declare_partials(
            of="*",
            wrt=["true_airspeed", "dynamic_viscosity"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        air_velocity = inputs["true_airspeed"]
        fin_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_spacing"
        ]
        fin_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_length"
        ]
        mean_air_dynamic_viscosity = inputs["dynamic_viscosity"]

        outputs["reynolds_number"] = (
            fin_spacing**2.0 * air_velocity / (mean_air_dynamic_viscosity * fin_length)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        air_velocity = inputs["true_airspeed"]
        fin_spacing = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_spacing"
        ]
        fin_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_length"
        ]
        mean_air_dynamic_viscosity = inputs["dynamic_viscosity"]

        partials[
            "reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_spacing",
        ] = fin_spacing * 2.0 * air_velocity / (mean_air_dynamic_viscosity * fin_length)

        partials["reynolds_number", "dynamic_viscosity"] = (
            -(fin_spacing**2.0) * air_velocity / (mean_air_dynamic_viscosity**2.0 * fin_length)
        )

        partials["reynolds_number", "true_airspeed"] = fin_spacing**2.0 / (
            mean_air_dynamic_viscosity * fin_length
        )

        partials[
            "reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":fin_length",
        ] = -(fin_spacing**2.0) * air_velocity / (mean_air_dynamic_viscosity * fin_length**2.0)
