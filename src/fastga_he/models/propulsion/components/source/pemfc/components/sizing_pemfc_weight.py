# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import SUBMODEL_SIZING_PEMFC_WEIGHT
import fastoad.api as oad

FC_WEIGHT_DENSITY = 8.5034  # kg/m^2
DEFAULT_FC_SPECIFIC_POWER = 0.345  # kW/kg

oad.RegisterSubmodel.active_models[SUBMODEL_SIZING_PEMFC_WEIGHT] = (
    "fastga_he.submodel.propulsion.sizing.pemfc.weight.base"
)


@oad.RegisterSubmodel(
    SUBMODEL_SIZING_PEMFC_WEIGHT,
    "fastga_he.submodel.propulsion.sizing.pemfc.weight.base",
)
class SizingPEMFCWeightAerostak200W(om.ExplicitComponent):
    """
    Computation of the weight the PEMFC based on the layer weight density of the Aerostak 200W stack.
    """

    # TODO: Edit citation after rebase
    # TODO: Adding another way of mass estimation from D.Juschus
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
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Number of layer in 1 PEMFC stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            units="m**2",
            val=np.nan,
            desc="Effective fuel cell area in the stack",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            units="kg",
            val=500.0,
            desc="Mass of the pemfc stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass"] = (
            FC_WEIGHT_DENSITY
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = (
            FC_WEIGHT_DENSITY
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
        ] = (
            FC_WEIGHT_DENSITY
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
        )


@oad.RegisterSubmodel(
    SUBMODEL_SIZING_PEMFC_WEIGHT,
    "fastga_he.submodel.propulsion.sizing.pemfc.weight.adjusted",
)
class SizingPEMFCWeightAdjusted(om.ExplicitComponent):
    """
    Computation of the weight the PEMFC based on the layer weight density but adjusted with power density
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
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Number of layer in 1 PEMFC stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            units="m**2",
            val=np.nan,
            desc="Effective fuel cell area in the stack",
        )

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":max_specific_power",
            units="kW/kg",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            units="kg",
            val=500.0,
            desc="Mass of the pemfc stack",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        adjust_factor = (
            DEFAULT_FC_SPECIFIC_POWER
            / inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_specific_power"
            ]
        )

        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass"] = (
            FC_WEIGHT_DENSITY
            * adjust_factor
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        adjust_factor = (
            DEFAULT_FC_SPECIFIC_POWER
            / inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_specific_power"
            ]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = (
            FC_WEIGHT_DENSITY
            * adjust_factor
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
        ] = (
            FC_WEIGHT_DENSITY
            * adjust_factor
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
        )

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_specific_power",
        ] = (
            -FC_WEIGHT_DENSITY
            * DEFAULT_FC_SPECIFIC_POWER
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
            / inputs[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_specific_power"
            ]
            ** 2
        )
