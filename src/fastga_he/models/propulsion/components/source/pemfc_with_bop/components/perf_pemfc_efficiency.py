# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import (
    HHV_HYDROGEN_EQUIVALENT_VOLTAGE,
    FUEL_UTILIZATION_COEFFICIENT,
    DEFAULT_PEMFC_EFFICIENCY,
)


class PerformancesPEMFCStackBOPEfficiency(om.ExplicitComponent):
    """
    Efficiency computation  of the PEMFC with dividing the actual voltage provided by the
    fuel cell with the higher heating value (HHV) of hydrogen. The convertion into voltage form
    is simply calculated by dividing the HHV of hydrogen (285.5 kJ/mol) by the amount of
    electrons produced by single hydrogen particle and Faraday's constant.
    source: https://www.nrel.gov/docs/fy10osti/47302.pdf
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

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
            val=np.nan,
            desc="Total number of layers in the PEMFC stack",
        )
        self.add_input("fuel_cell_voltage", units="V", val=np.full(number_of_points, np.nan))

        self.add_output(
            name="efficiency",
            val=np.full(number_of_points, DEFAULT_PEMFC_EFFICIENCY),
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            of="efficiency",
            wrt=["single_layer_pemfc_voltage", "fuel_cell_voltage"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="efficiency",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        single_layer_pemfc_voltage = inputs["single_layer_pemfc_voltage"]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers"
        ]
        fuel_cell_voltage = inputs["fuel_cell_voltage"]

        clipped_efficiency = np.clip(
            (single_layer_pemfc_voltage - fuel_cell_voltage / number_of_layers)
            * FUEL_UTILIZATION_COEFFICIENT
            / HHV_HYDROGEN_EQUIVALENT_VOLTAGE,
            0.1,
            0.7,
        )

        outputs["efficiency"] = clipped_efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        single_layer_pemfc_voltage = inputs["single_layer_pemfc_voltage"]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers"
        ]
        fuel_cell_voltage = inputs["fuel_cell_voltage"]

        efficiency = (
            (single_layer_pemfc_voltage - fuel_cell_voltage / number_of_layers)
            * FUEL_UTILIZATION_COEFFICIENT
            / HHV_HYDROGEN_EQUIVALENT_VOLTAGE
        )

        clipped_efficiency = np.clip(efficiency, 0.1, 0.7)

        partials["efficiency", "single_layer_pemfc_voltage"] = np.clip(
            efficiency == clipped_efficiency,
            FUEL_UTILIZATION_COEFFICIENT / HHV_HYDROGEN_EQUIVALENT_VOLTAGE,
            1e-6,
        )

        partials["efficiency", "fuel_cell_voltage"] = np.clip(
            efficiency == clipped_efficiency,
            -FUEL_UTILIZATION_COEFFICIENT / (number_of_layers * HHV_HYDROGEN_EQUIVALENT_VOLTAGE),
            1e-6,
        )

        partials[
            "efficiency",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
        ] = np.clip(
            efficiency == clipped_efficiency,
            fuel_cell_voltage
            * FUEL_UTILIZATION_COEFFICIENT
            / (number_of_layers**2.0 * HHV_HYDROGEN_EQUIVALENT_VOLTAGE),
            1e-6,
        )
