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

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the output side of the converter",
        )
        self.add_input(
            "voltage_out_target",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Target voltage at the output side of the converter",
        )
        self.add_input(
            "dc_current_out",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current at the output side of the converter",
        )
        self.add_input(
            "efficiency",
            val=np.full(number_of_points, np.nan),
            desc="Efficiency of the converter",
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
            of="power_rel",
            wrt=["dc_voltage_out", "dc_current_out", "power_rel", "efficiency"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="voltage_out_rel",
            wrt="voltage_out_rel",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )
        self.declare_partials(
            of="voltage_out_rel",
            wrt="voltage_out_target",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=-np.ones(number_of_points),
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        residuals["voltage_out_rel"] = outputs["voltage_out_rel"] - inputs["voltage_out_target"]
        residuals["power_rel"] = (inputs["dc_voltage_out"] * inputs["dc_current_out"]) - outputs[
            "power_rel"
        ] * inputs["efficiency"]

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        jacobian["power_rel", "power_rel"] = -inputs["efficiency"]
        jacobian["power_rel", "efficiency"] = -outputs["power_rel"]
        jacobian["power_rel", "dc_voltage_out"] = inputs["dc_current_out"]
        jacobian["power_rel", "dc_current_out"] = inputs["dc_voltage_out"]
