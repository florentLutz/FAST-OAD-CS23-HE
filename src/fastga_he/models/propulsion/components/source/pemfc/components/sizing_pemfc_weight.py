# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

CELL_DENSITY = 8.5034  # [kg/m^2]
DEFAULT_FC_SPECIFIC_POWER = 0.345  # [kW/kg]


class SizingPEMFCStackWeight(om.ExplicitComponent):
    """
    Computation of the weight the PEMFC based on the layer weight density but adjusted with
    power density. The calculation is based on the equations given by :cite:`hoogendoorn:2018`.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Number of layer in 1 PEMFC stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="m**2",
            val=np.nan,
            desc="Effective fuel cell area in the stack",
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
            desc="Mass of PEMFC stack",
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
        specific_power_ratio = DEFAULT_FC_SPECIFIC_POWER / specific_power

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass"] = (
            CELL_DENSITY * specific_power_ratio * effective_area * number_of_layers
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
        specific_power_ratio = DEFAULT_FC_SPECIFIC_POWER / specific_power

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = CELL_DENSITY * specific_power_ratio * effective_area

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
        ] = CELL_DENSITY * specific_power_ratio * number_of_layers

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":specific_power",
        ] = (
            -CELL_DENSITY
            * DEFAULT_FC_SPECIFIC_POWER
            * number_of_layers
            * effective_area
            / specific_power**2
        )
