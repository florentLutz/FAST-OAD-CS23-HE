# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class that identifies the maximum voltage and current of the bus bar in order to size it.
    """

    def initialize(self):

        self.options.declare(
            name="ac_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="number_of_inputs",
            default=1,
            types=int,
            desc="Number of connections at the input of the bus",
            allow_none=False,
        )

    def setup(self):

        ac_bus_id = self.options["ac_bus_id"]
        number_of_points = self.options["number_of_points"]

        # TODO: replace with peak value
        self.add_input(
            name="ac_voltage_peak",
            val=np.full(number_of_points, np.nan),
            units="V",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":voltage_ac_max",
            units="V",
            val=800.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":voltage_ac_max",
            wrt="ac_voltage_peak",
            method="exact",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":current_ac_max",
            units="A",
            val=500.0,
            desc="Maximum RMS current flowing through one phase of the bus",
        )

        # For once, input are going to be after outputs, just to ensure the declaration of
        # partials goes well
        for i in range(self.options["number_of_inputs"]):
            self.add_input(
                name="ac_current_rms_in_one_phase_" + str(i + 1),
                units="A",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="RMS value of the current going through one phase of the bus at output number "
                + str(j + 1),
            )

            self.declare_partials(
                of="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":current_ac_max",
                wrt="ac_current_rms_in_one_phase_" + str(i + 1),
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ac_bus_id = self.options["ac_bus_id"]
        number_of_points = self.options["number_of_points"]

        outputs["data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":voltage_ac_max"] = np.max(
            inputs["ac_voltage_peak"]
        )

        total_current = np.zeros(number_of_points)
        for i in range(self.options["number_of_inputs"]):
            total_current += inputs["ac_current_rms_in_one_phase_" + str(i + 1)]

        outputs["data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":current_ac_max"] = np.max(
            total_current
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ac_bus_id = self.options["ac_bus_id"]
        number_of_points = self.options["number_of_points"]

        ac_voltage_peak = inputs["ac_voltage_peak"]

        partials[
            "data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":voltage_ac_max",
            "ac_voltage_peak",
        ] = np.where(ac_voltage_peak == np.amax(ac_voltage_peak), 1.0, 0.0)

        total_current = np.zeros(number_of_points)
        for i in range(self.options["number_of_inputs"]):
            total_current += inputs["dc_current_in_" + str(i + 1)]

        for j in range(self.options["number_of_inputs"]):
            partials[
                "data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":current_ac_max",
                "dc_current_in_" + str(j + 1),
            ] = np.where(total_current == np.amax(total_current), 1.0, 0.0)
