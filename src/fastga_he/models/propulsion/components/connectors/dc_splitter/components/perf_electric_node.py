# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesElectricalNode(om.ImplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="dc_current_out",
            units="A",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Current going out of the bus at the output",
        )
        self.add_input(
            name="dc_current_in_1",
            units="A",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Current going in the bus at the primary input (number 1)",
        )
        self.add_input(
            name="dc_current_in_2",
            units="A",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Current going in the bus at the primary input (number 2)",
        )
        self.add_input(
            "power_split",
            units="percent",
            val=np.nan,
            shape=number_of_points,
            desc="Percent of the power going to the first (primary) input, in %, with a format "
            "adapted to mission",
        )

        self.add_output(
            name="dc_voltage",
            val=np.full(number_of_points, 350),
            units="V",
            desc="Voltage of the bus",
            lower=np.full(number_of_points, 0.0),
            upper=np.full(number_of_points, 2.0e3),
        )
        self.add_output(
            name="dc_voltage_in_1",
            val=np.full(number_of_points, 800.0),
            units="V",
            lower=np.full(number_of_points, 0.0),
            upper=np.full(number_of_points, 2.0e3),
        )
        self.add_output(
            name="dc_voltage_in_2",
            val=np.full(number_of_points, 800.0),
            units="V",
            lower=np.full(number_of_points, 0.0),
            upper=np.full(number_of_points, 2.0e3),
        )

        self.declare_partials(
            of="dc_voltage",
            wrt=["dc_voltage_in_1", "dc_voltage_in_2", "dc_voltage"],
            method="exact",
        )
        self.declare_partials(
            of="dc_voltage_in_1",
            wrt=["dc_current_in_1", "dc_current_out", "power_split"],
            method="exact",
        )
        self.declare_partials(
            of="dc_voltage_in_2",
            wrt=["dc_current_in_1", "dc_current_out", "dc_current_in_2"],
            method="exact",
        )

    def apply_nonlinear(self, inputs, outputs, residuals):

        dc_current_out = inputs["dc_current_out"]

        dc_current_in_1 = inputs["dc_current_in_1"]
        dc_current_in_2 = inputs["dc_current_in_2"]

        power_split = inputs["power_split"]

        dc_voltage = outputs["dc_voltage"]

        dc_voltage_in_1 = outputs["dc_voltage_in_1"]
        dc_voltage_in_2 = outputs["dc_voltage_in_2"]

        residuals["dc_voltage"] = (dc_voltage_in_1 + dc_voltage_in_2) / 2 - dc_voltage
        residuals["dc_voltage_in_1"] = dc_current_in_1 - power_split * dc_current_out / 100.0
        residuals["dc_voltage_in_2"] = dc_current_in_1 + dc_current_in_2 - dc_current_out

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        partials["dc_voltage", "dc_voltage_in_1"] = np.eye(number_of_points) / 2.0
        partials["dc_voltage", "dc_voltage_in_2"] = np.eye(number_of_points) / 2.0
        partials["dc_voltage", "dc_voltage"] = -np.eye(number_of_points)

        partials["dc_voltage_in_1", "dc_current_in_1"] = np.eye(number_of_points)
        partials["dc_voltage_in_1", "power_split"] = -np.diag(inputs["dc_current_out"] / 100.0)
        partials["dc_voltage_in_1", "dc_current_out"] = -np.diag(inputs["power_split"] / 100.0)

        partials["dc_voltage_in_2", "dc_current_in_1"] = np.eye(number_of_points)
        partials["dc_voltage_in_2", "dc_current_in_2"] = np.eye(number_of_points)
        partials["dc_voltage_in_2", "dc_current_out"] = -np.eye(number_of_points)
