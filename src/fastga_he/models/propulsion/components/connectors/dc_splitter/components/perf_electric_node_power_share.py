# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesElectricalNodePowerShare(om.ImplicitComponent):
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
            "power_share",
            units="W",
            val=np.nan,
            shape=number_of_points,
            desc="Share of the power going to the first (primary) input, in W, with a format "
            "adapted to mission. If below nothing will go in the secondary input, if above, "
            "the complement will flow in the secondary input",
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
            wrt=["dc_current_in_1", "dc_voltage_in_1", "power_share"],
            method="exact",
        )
        self.declare_partials(
            of="dc_voltage_in_2",
            wrt=[
                "dc_current_in_2",
                "dc_voltage_in_2",
                "dc_current_out",
                "dc_voltage",
                "power_share",
            ],
            method="exact",
        )

    def apply_nonlinear(self, inputs, outputs, residuals):

        dc_current_out = inputs["dc_current_out"]

        dc_current_in_1 = inputs["dc_current_in_1"]
        dc_current_in_2 = inputs["dc_current_in_2"]

        power_share = inputs["power_share"]

        dc_voltage = outputs["dc_voltage"]

        dc_voltage_in_1 = outputs["dc_voltage_in_1"]
        dc_voltage_in_2 = outputs["dc_voltage_in_2"]

        power_output = dc_voltage * dc_current_out

        # The multiplication by 1e3 are there so that all the residuals have more or less the
        # same order of magnitude, the third one being hundreds of kW, the other hundreds of V/A.
        # And why not divide you might ask ? Well, by trial and error on the splitter assembly,
        # it seems to work best this way \_(:o)_/

        residuals["dc_voltage"] = ((dc_voltage_in_1 + dc_voltage_in_2) / 2 - dc_voltage) * 1e3
        residuals["dc_voltage_in_1"] = (dc_current_in_1 * dc_voltage_in_1 - power_share) * 1e3

        residuals_dc_v_in_2_below_power_share = dc_current_in_2 * 1e3
        residuals_dc_v_in_2_above_power_share = (
            power_output - power_share
        ) - dc_current_in_2 * dc_voltage_in_2

        residuals["dc_voltage_in_2"] = np.where(
            power_output < power_share,
            residuals_dc_v_in_2_below_power_share,
            residuals_dc_v_in_2_above_power_share,
        )

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        dc_current_out = inputs["dc_current_out"]
        dc_voltage = outputs["dc_voltage"]

        power_share = inputs["power_share"]

        power_output = dc_voltage * dc_current_out

        partials["dc_voltage", "dc_voltage_in_1"] = np.eye(number_of_points) / 2.0 * 1e3
        partials["dc_voltage", "dc_voltage_in_2"] = np.eye(number_of_points) / 2.0 * 1e3
        partials["dc_voltage", "dc_voltage"] = -np.eye(number_of_points) * 1e3

        partials["dc_voltage_in_1", "dc_current_in_1"] = np.diag(outputs["dc_voltage_in_1"] * 1e3)
        partials["dc_voltage_in_1", "power_share"] = -np.eye(number_of_points) * 1e3
        partials["dc_voltage_in_1", "dc_voltage_in_1"] = np.diag(inputs["dc_current_in_1"] * 1e3)

        partials_dc_v_in_2_below = np.zeros(number_of_points) * 1e3
        partials_dc_v_in_2_above = -inputs["dc_current_in_2"]

        partials_dc_a_in_2_below = np.ones(number_of_points) * 1e3
        partials_dc_a_in_2_above = -outputs["dc_voltage_in_2"]

        partials_dc_a_out_below = np.zeros(number_of_points)
        partials_dc_a_out_above = outputs["dc_voltage"]

        partials_dc_v_out_below = np.zeros(number_of_points)
        partials_dc_v_out_above = inputs["dc_current_out"]

        partials_power_share_below = np.zeros(number_of_points)
        partials_power_share_above = -np.ones(number_of_points)

        partials["dc_voltage_in_2", "dc_voltage_in_2"] = np.diag(
            np.where(power_output < power_share, partials_dc_v_in_2_below, partials_dc_v_in_2_above)
        )
        partials["dc_voltage_in_2", "dc_current_in_2"] = np.diag(
            np.where(power_output < power_share, partials_dc_a_in_2_below, partials_dc_a_in_2_above)
        )
        partials["dc_voltage_in_2", "dc_current_out"] = np.diag(
            np.where(power_output < power_share, partials_dc_a_out_below, partials_dc_a_out_above)
        )
        partials["dc_voltage_in_2", "dc_voltage"] = np.diag(
            np.where(power_output < power_share, partials_dc_v_out_below, partials_dc_v_out_above)
        )
        partials["dc_voltage_in_2", "power_share"] = np.diag(
            np.where(
                power_output < power_share, partials_power_share_below, partials_power_share_above
            )
        )
