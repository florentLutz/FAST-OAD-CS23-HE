# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum currents, voltage, torque and rpm of the generator.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):

        generator_id = self.options["generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_out_one_phase",
            units="A",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            "ac_voltage_peak_out",
            units="V",
            val=np.full(number_of_points, np.nan),
            desc="Peak line to neutral voltage at the input of the generator",
        )
        self.add_input("torque_in", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("power_losses", units="W", val=np.nan, shape=number_of_points)
        self.add_input("shaft_power_in", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":current_ac_max",
            units="A",
            val=600.0,
            desc="Maximum value of the RMS current flowing through one phase of the generator",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":current_ac_max",
            wrt="ac_current_rms_out_one_phase",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":voltage_ac_max",
            units="V",
            val=800.0,
            desc="Maximum value of the peak voltage at the output of the generator",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":voltage_ac_max",
            wrt="ac_voltage_peak_out",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_max",
            units="N*m",
            val=1000.0,
            desc="Maximum value of the torque the generator receives",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":torque_max",
            wrt="torque_in",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":rpm_max",
            units="min**-1",
            val=5000.0,
            desc="Maximum value of the generator rpm",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":rpm_max",
            wrt="rpm",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":losses_max",
            units="W",
            val=4200.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":losses_max",
            wrt="power_losses",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max",
            units="W",
            val=42000.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max",
            wrt="shaft_power_in",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        generator_id = self.options["generator_id"]

        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":current_ac_max"
        ] = np.max(inputs["ac_current_rms_out_one_phase"])
        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":voltage_ac_max"
        ] = np.max(inputs["ac_voltage_peak_out"])
        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_max"
        ] = np.max(inputs["torque_in"])
        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":rpm_max"] = np.max(
            inputs["rpm"]
        )
        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":losses_max"
        ] = np.max(inputs["power_losses"])
        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max"
        ] = np.max(inputs["shaft_power_in"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        generator_id = self.options["generator_id"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":current_ac_max",
            "ac_current_rms_out_one_phase",
        ] = np.where(
            inputs["ac_current_rms_out_one_phase"]
            == np.max(inputs["ac_current_rms_out_one_phase"]),
            1.0,
            0.0,
        )
        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":voltage_ac_max",
            "ac_voltage_peak_out",
        ] = np.where(
            inputs["ac_voltage_peak_out"] == np.max(inputs["ac_voltage_peak_out"]), 1.0, 0.0
        )
        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_max", "torque_in"
        ] = np.where(inputs["torque_in"] == np.max(inputs["torque_in"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":rpm_max", "rpm"
        ] = np.where(inputs["rpm"] == np.max(inputs["rpm"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":losses_max",
            "power_losses",
        ] = np.where(inputs["power_losses"] == np.max(inputs["power_losses"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":shaft_power_max",
            "shaft_power_in",
        ] = np.where(inputs["shaft_power_in"] == np.max(inputs["shaft_power_in"]), 1.0, 0.0)
