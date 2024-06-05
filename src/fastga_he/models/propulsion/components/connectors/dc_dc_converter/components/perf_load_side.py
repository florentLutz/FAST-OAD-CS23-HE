# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesConverterLoadSide(om.ExplicitComponent):
    """
    The converter is divided between a load side where the battery is and a generator side where
    the rest of the circuit is. This component represents the load side.

    Based on the methodology from :cite:`hendricks:2019`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        batman = np.full(number_of_points, np.nan)

        self.add_input(
            "dc_voltage_in",
            val=batman,
            units="V",
            desc="Voltage at the input side of the converter",
        )
        self.add_input(
            "power", val=batman, units="W", desc="Power at the input side of the converter"
        )

        self.add_output(
            "dc_current_in",
            val=np.full(number_of_points, 400.0),
            units="A",
            desc="Current at the input side of the converter",
            lower=-1000.0,
            upper=1000.0,
        )

        self.declare_partials(
            of="dc_current_in",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # If power is too low, we consider that there is no current, may prevent some convergence
        # issue
        current_in = np.where(
            np.abs(inputs["power"]) < 10.0, 0.0, inputs["power"] / inputs["dc_voltage_in"]
        )

        outputs["dc_current_in"] = current_in

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials_voltage = np.where(
            np.abs(inputs["power"]) < 10.0, 1e-8, -inputs["power"] / inputs["dc_voltage_in"] ** 2
        )
        partials["dc_current_in", "dc_voltage_in"] = partials_voltage

        partials_power = np.where(
            np.abs(inputs["power"]) < 10.0, 1e-8, 1.0 / inputs["dc_voltage_in"]
        )
        partials["dc_current_in", "power"] = partials_power
