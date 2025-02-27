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


class PerformancesPEMFCStackEfficiency(om.ExplicitComponent):
    """
    Computation of efficiency of the PEMFC  with dividing the actual voltage provided by the
    fuel cell with the higher heating value (HHV) of hydrogen. The convertion into voltage form
    is simply calculated by dividing the HHV of hydrogen (285.5 kJ/mol) by the amount of
    electrons produced by single hydrogen particle and Faraday's constant.
    source: https://www.nrel.gov/docs/fy10osti/47302.pdf
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            name="efficiency",
            val=np.full(number_of_points, DEFAULT_PEMFC_EFFICIENCY),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.full(
                number_of_points, FUEL_UTILIZATION_COEFFICIENT / HHV_HYDROGEN_EQUIVALENT_VOLTAGE
            ),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["efficiency"] = (
            inputs["single_layer_pemfc_voltage"]
            * FUEL_UTILIZATION_COEFFICIENT
            / HHV_HYDROGEN_EQUIVALENT_VOLTAGE
        )
