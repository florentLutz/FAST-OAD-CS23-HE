# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesConverterRelations(om.ImplicitComponent):
    """
    The converter is divided between a load side where the battery is and a generator side where
    the rest of the circuit is. This component represents the link between the two sides.

    Based on the methodology from :cite:`hendricks:2019`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "voltage_target", default=1000, desc="Target output voltage of the converter"
        )
        self.options.declare("efficiency", default=0.98, desc="Efficiency of the converter")

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="voltage to output side",
        )
        self.add_input(
            "current_out",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="current to output side",
        )

        self.add_output(
            "voltage_out_rel",
            val=np.full(number_of_points, 500.0),
            units="V",
            desc="Voltage of the output side in the relations equation",
        )
        self.add_output(
            "power_rel",
            val=np.full(number_of_points, 300.0e3),
            units="W",
            desc="Power in the relations equation",
        )

        self.declare_partials(
            of="power_rel", wrt=["voltage_out", "current_out", "power_rel"], method="exact"
        )
        self.declare_partials(of="voltage_out_rel", wrt="voltage_out_rel", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["voltage_out_rel"] = outputs["voltage_out_rel"] - self.options["voltage_target"]
        residuals["power_rel"] = (inputs["voltage_out"] * inputs["current_out"]) - outputs[
            "power_rel"
        ] * self.options["efficiency"]

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        partials["power_rel", "power_rel"] = -np.eye(number_of_points) * self.options["efficiency"]
        partials["power_rel", "voltage_out"] = np.diag(inputs["current_out"])
        partials["power_rel", "current_out"] = np.diag(inputs["voltage_out"])

        partials["voltage_out_rel", "voltage_out_rel"] = np.eye(number_of_points)
