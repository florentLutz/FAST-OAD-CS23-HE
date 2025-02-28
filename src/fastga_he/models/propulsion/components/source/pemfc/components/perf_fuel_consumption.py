# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import (
    FARADAY_CONSTANT,
    H2_MOL_PER_KG,
    NUMBER_OF_ELETRONS_FROM_H2,
    DEFAULT_HYDROGEN_CONSUMPTION,
)


class PerformancesPEMFCStackFuelConsumption(om.ExplicitComponent):
    """
    Computation of the hydrogen consumption for the required power. Simply based on the
    results of the number of layers, current density, and effective area of the PEMFC stack.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=np.nan,
            desc="Effective area of the PEMFC's polymer electrolyte membrane",
        )

        self.add_input(
            "fc_current_density",
            val=np.nan,
            shape=number_of_points,
            units="A/cm**2",
            desc="Current density of the PEMFC stack",
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Total number of layers in the PEMFC stack",
        )

        self.add_output(
            "fuel_consumption",
            units="kg/h",
            val=DEFAULT_HYDROGEN_CONSUMPTION,
            shape=number_of_points,
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":number_of_layers",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="fc_current_density",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["fuel_consumption"] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
            * inputs["fc_current_density"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
            * 3600
            / (NUMBER_OF_ELETRONS_FROM_H2 * FARADAY_CONSTANT * H2_MOL_PER_KG)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "fuel_consumption",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = (
            inputs["fc_current_density"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
            * 3600
            / (NUMBER_OF_ELETRONS_FROM_H2 * FARADAY_CONSTANT * H2_MOL_PER_KG)
        )
        partials["fuel_consumption", "fc_current_density"] = (
            np.ones(number_of_points)
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
            * 3600
            / (NUMBER_OF_ELETRONS_FROM_H2 * FARADAY_CONSTANT * H2_MOL_PER_KG)
        )
        partials[
            "fuel_consumption",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
        ] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
            * inputs["fc_current_density"]
            * 3600
            / (NUMBER_OF_ELETRONS_FROM_H2 * FARADAY_CONSTANT * H2_MOL_PER_KG)
        )
