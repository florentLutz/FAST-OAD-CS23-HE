# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

CELL_DENSITY = 8.5034  # [kg/m^2]
DEFAULT_FC_SPECIFIC_POWER = 0.345  # [kW/kg]


class SizingPEMFCStackWeight(om.ExplicitComponent):
    """
    Computation of the PEMFC stack weight based on the layer weight density, adjusted with
    specific power. The calculation is based on the equations given by :cite:`hoogendoorn:2018`.
    This computation consider the weight of the PEMFC stack and the BoPs of the PEMFC.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Total number of layers in the PEMFC stack",
        )

        self.add_input(
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_mass",
            val=1.0,
            desc="A tuning factor allows the PEMFC stack weight to be changed manually.",
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="m**2",
            val=np.nan,
            desc="Effective area of the PEMFC's polymer electrolyte membrane",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power",
            units="kW/kg",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            units="kg",
            val=500.0,
            desc="Mass of the PEMFC stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        specific_power = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power"
        ]
        effective_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
        ]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
        ]

        k_mass = inputs[
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_mass"
        ]

        specific_power_ratio = DEFAULT_FC_SPECIFIC_POWER / specific_power

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass"] = (
            k_mass * CELL_DENSITY * specific_power_ratio * effective_area * number_of_layers
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        specific_power = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power"
        ]
        effective_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
        ]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
        ]
        k_mass = inputs[
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_mass"
        ]

        specific_power_ratio = DEFAULT_FC_SPECIFIC_POWER / specific_power

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = k_mass * CELL_DENSITY * specific_power_ratio * effective_area

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
        ] = k_mass * CELL_DENSITY * specific_power_ratio * number_of_layers

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power",
        ] = (
            -k_mass
            * CELL_DENSITY
            * DEFAULT_FC_SPECIFIC_POWER
            * number_of_layers
            * effective_area
            / specific_power**2
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            "settings:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":k_mass",
        ] = CELL_DENSITY * specific_power_ratio * effective_area * number_of_layers
