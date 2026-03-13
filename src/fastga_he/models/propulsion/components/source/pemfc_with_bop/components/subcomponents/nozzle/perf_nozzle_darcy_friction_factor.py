# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleDarcyFrictionFactor(om.ExplicitComponent):
    """
    Computation of the Darcy flow friction factor in the nozzle.
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
            desc="Identifier of the diffuser",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_input(
            "air_reynolds_number",
            val=0.3,
            units="unitless",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":relative_roughness",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "nozzle_darcy_friction_factor",
            val=0.3,
            units="unitless",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":relative_roughness",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        reynolds_number = inputs["air_reynolds_number"]
        relative_roughness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":relative_roughness"
        ]

        if reynolds_number < 3000.0:
            outputs["nozzle_darcy_friction_factor"] = 64.0 / reynolds_number
        else:
            # Haaland's equation
            outputs["nozzle_darcy_friction_factor"] = (
                -1.8 * np.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number)
            ) ** -2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        reynolds_number = inputs["air_reynolds_number"]
        relative_roughness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":relative_roughness"
        ]

        partials[
            "nozzle_darcy_friction_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":relative_roughness",
        ] = np.where(
            reynolds_number < 3000.0,
            0.0,
            2.0
            * (-1.8 * np.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number) ** -3.0)
            * (1.8 / np.log(10))
            * (1.11 / relative_roughness * (relative_roughness / 3.7) ** 1.11)
            / (3.7 * ((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number)),
        )

        partials["nozzle_darcy_friction_factor", "air_reynolds_number"] = np.where(
            reynolds_number < 3000.0,
            -64.0 / reynolds_number**2,
            2.0
            * (-1.8 * np.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number) ** -3.0)
            * (1.8 / np.log(10))
            * (6.9 / reynolds_number**2)
            / ((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number),
        )
