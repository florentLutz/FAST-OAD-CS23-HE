# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirReynoldsNumber(om.ExplicitComponent):
    """
    Computation of the air Reynolds number in the heat exchanger.
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
            name="air_mass_velocity",
            units="kg/s/m**2",
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
            + ":mean_air_dynamic_viscosity",
            val=np.nan,
            units="Pa*s",
        )

        self.add_output(
            name="air_reynolds_number",
            units="unitless",
            val=2400.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_mass_velocity = inputs["air_mass_velocity"]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
        ]
        mean_air_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_dynamic_viscosity"
        ]

        outputs["air_reynolds_number"] = (
            air_mass_velocity * fin_hydraulic_diameter / mean_air_dynamic_viscosity
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["heat_exchanger_id"]

        air_mass_velocity = inputs["air_mass_velocity"]
        fin_hydraulic_diameter = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter"
        ]
        mean_air_dynamic_viscosity = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_dynamic_viscosity"
        ]

        partials["air_reynolds_number", "air_mass_velocity"] = (
            fin_hydraulic_diameter / mean_air_dynamic_viscosity
        )

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":fin_hydraulic_diameter",
        ] = air_mass_velocity / mean_air_dynamic_viscosity

        partials[
            "air_reynolds_number",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":mean_air_dynamic_viscosity",
        ] = -air_mass_velocity * fin_hydraulic_diameter / mean_air_dynamic_viscosity**2.0
