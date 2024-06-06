# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum currents (DC and AC 1 phase), voltage (DC and AC peak) of the
    rectifier
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_current_out",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current at the output side of the rectifier",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the output side of the rectifier",
        )

        self.add_input(
            "ac_current_rms_in_one_phase",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current at the input side of the rectifier",
        )
        self.add_input(
            "ac_voltage_peak_in",
            units="V",
            val=np.full(number_of_points, np.nan),
            desc="Peak line to neutral voltage at the input of the rectifier",
        )

        self.add_input("switching_frequency", units="Hz", val=np.nan, shape=number_of_points)
        self.add_input(
            "losses_rectifier",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max",
            units="A",
            val=250.0,
            desc="Maximum RMS current flowing through one arm of the rectifier",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max",
            wrt="ac_current_rms_in_one_phase",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max",
            units="V",
            val=800.0,
            desc="Maximum value of the peak voltage at the input of the rectifier (AC side)",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max",
            wrt="ac_voltage_peak_in",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_dc_max",
            units="A",
            val=750.0,
            desc="Maximum value of the current at the output of the rectifier",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_dc_max",
            wrt="dc_current_out",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_dc_max",
            units="V",
            val=800,
            desc="Maximum value of the voltage at the output of the rectifier (DC side)",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_dc_max",
            wrt="dc_voltage_out",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency_max",
            units="Hz",
            val=15.0e3,
            desc="Maximum value of the switching frequency of the rectifier",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency_max",
            wrt="switching_frequency",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max",
            units="W",
            val=42.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max",
            wrt="losses_rectifier",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max"
        ] = np.max(inputs["ac_current_rms_in_one_phase"])
        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max"
        ] = np.max(inputs["ac_voltage_peak_in"])
        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_dc_max"
        ] = np.max(inputs["dc_current_out"])
        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_dc_max"
        ] = np.max(inputs["dc_voltage_out"])
        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency_max"
        ] = np.max(inputs["switching_frequency"])
        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max"
        ] = np.max(inputs["losses_rectifier"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rectifier_id = self.options["rectifier_id"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_max",
            "ac_current_rms_in_one_phase",
        ] = np.where(
            inputs["ac_current_rms_in_one_phase"] == np.max(inputs["ac_current_rms_in_one_phase"]),
            1.0,
            0.0,
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_max",
            "ac_voltage_peak_in",
        ] = np.where(inputs["ac_voltage_peak_in"] == np.max(inputs["ac_voltage_peak_in"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_dc_max",
            "dc_current_out",
        ] = np.where(inputs["dc_current_out"] == np.max(inputs["dc_current_out"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_dc_max",
            "dc_voltage_out",
        ] = np.where(inputs["dc_voltage_out"] == np.max(inputs["dc_voltage_out"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency_max",
            "switching_frequency",
        ] = np.where(
            inputs["switching_frequency"] == np.max(inputs["switching_frequency"]), 1.0, 0.0
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":losses_max",
            "losses_rectifier",
        ] = np.where(inputs["losses_rectifier"] == np.max(inputs["losses_rectifier"]), 1.0, 0.0)
