# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

DEFAULT_PEMFC_EFFICIENCY = 0.53
DEFAULT_PRESSURE_ATM = 1.0


class PerformancesPEMFCEfficiency(om.ExplicitComponent):
    """
    Computation of efficiency of the battery based on the losses at battery level and the output
    voltage and current.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            "pressure coefficient", default=0.06, desc="pressure coefficient of one layer of pemfc"
        )

        self.options.declare(
            "pemfc_theoretical_electric_potential",
            default=1.229,
            desc="pemfc_theoretical_electric_potential of one layer of pemfc [V]",
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "nominal_pressure",
            units="atm",
            val=DEFAULT_PRESSURE_ATM,
        )

        self.add_input(
            "operation_pressure",
            units="atm",
            val=np.full(number_of_points, DEFAULT_PRESSURE_ATM),
        )

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
            wrt=["single_layer_pemfc_voltage", "operation_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="nominal_pressure",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        e0 = self.options["pemfc_theoretical_electric_potential"]
        pressure_coeff = self.options["pressure coefficient"]
        operation_pressure = inputs["operation_pressure"]
        nominal_pressure = inputs["nominal_pressure"]

        e = e0 + pressure_coeff * np.log(operation_pressure / nominal_pressure)

        efficiency = inputs["single_layer_pemfc_voltage"] / e

        outputs["efficiency"] = efficiency

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]

        e0 = self.options["pemfc_theoretical_electric_potential"]
        pressure_coeff = self.options["pressure coefficient"]
        operation_pressure = inputs["operation_pressure"]
        nominal_pressure = inputs["nominal_pressure"]
        e = e0 + pressure_coeff * np.log(operation_pressure / nominal_pressure)

        partials["efficiency", "single_layer_pemfc_voltage",] = (
            np.ones(number_of_points) / e
        )

        partials["efficiency", "operation_pressure",] = (
            -pressure_coeff * inputs["single_layer_pemfc_voltage"] / (operation_pressure * e ** 2)
        )

        partials["efficiency", "nominal_pressure",] = (
            pressure_coeff * inputs["single_layer_pemfc_voltage"] / (nominal_pressure * e ** 2)
        )
