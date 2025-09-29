# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO


import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum currents, voltage, torque, losses, shaft power and rpm of the
    motor.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("ac_current_rms_in_one_phase", units="A", val=np.nan, shape=number_of_points)
        self.add_input(
            "ac_phase_current_density", units="A/m**2", val=np.nan, shape=number_of_points
        )
        self.add_input(
            "ac_voltage_peak_in",
            units="V",
            val=np.full(number_of_points, np.nan),
            desc="Peak line to neutral voltage at the input of the motor",
        )
        self.add_input("torque_out", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input("electromagnetic_torque", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("power_losses", units="W", val=np.nan, shape=number_of_points)
        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points)
        self.add_input(name="tangential_stress", units="MPa", val=np.nan, shape=number_of_points)
        self.add_input(
            name="surface_current_density", units="A/m", val=np.nan, shape=number_of_points
        )
        self.add_input(name="air_gap_flux_density", units="T", val=np.nan, shape=number_of_points)

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_ac_max",
            units="A",
            val=500.0,
            desc="Maximum value of the RMS current flowing through one phase of the motor",
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_max",
            units="A/m**2",
            val=1.0e4,
            desc="Maximum value of the RMS current flowing through one phase of the motor",
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_ac_max",
            units="V",
            val=800.0,
            desc="Maximum value of the peak voltage at the input of the motor",
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max",
            units="N*m",
            val=500.0,
            desc="Maximum value of the torque the motor has to provide",
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max",
            units="N*m",
            val=500.0,
            desc="Maximum value of the electromagnetic torque the motor has to provide",
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
            units="min**-1",
            val=5000.0,
            desc="Maximum value of the motor rpm",
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":losses_max",
            units="W",
            val=4200.0,
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_max",
            units="kW",
            val=4.20,
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_max",
            units="MPa",
            val=50.0,
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density_max",
            units="A/m",
            val=1.0,
        )
        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
            units="T",
            val=1.0,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_ac_max",
            wrt="ac_current_rms_in_one_phase",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_max",
            wrt="ac_phase_current_density",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_ac_max",
            wrt="ac_voltage_peak_in",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max",
            wrt="torque_out",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max",
            wrt="electromagnetic_torque",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
            wrt="rpm",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":losses_max",
            wrt="power_losses",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_max",
            wrt="shaft_power_out",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_max",
            wrt="tangential_stress",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":surface_current_density_max",
            wrt="surface_current_density",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
            wrt="air_gap_flux_density",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_ac_max"] = np.max(
            inputs["ac_current_rms_in_one_phase"]
        )
        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_max"
        ] = np.max(inputs["ac_phase_current_density"])
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_ac_max"] = np.max(
            inputs["ac_voltage_peak_in"]
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max"] = np.max(
            inputs["torque_out"]
        )
        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max"
        ] = np.max(inputs["electromagnetic_torque"])
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max"] = np.max(
            inputs["rpm"]
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":losses_max"] = np.max(
            inputs["power_losses"]
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_max"] = np.max(
            inputs["shaft_power_out"]
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_max"] = (
            np.max(inputs["tangential_stress"])
        )
        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density_max"
        ] = np.max(inputs["surface_current_density"])
        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max"
        ] = np.max(inputs["air_gap_flux_density"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_ac_max",
            "ac_current_rms_in_one_phase",
        ] = np.where(
            inputs["ac_current_rms_in_one_phase"] == np.max(inputs["ac_current_rms_in_one_phase"]),
            1.0,
            0.0,
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":current_density_ac_max",
            "ac_phase_current_density",
        ] = np.where(
            inputs["ac_phase_current_density"] == np.max(inputs["ac_phase_current_density"]),
            1.0,
            0.0,
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":voltage_ac_max",
            "ac_voltage_peak_in",
        ] = np.where(inputs["ac_voltage_peak_in"] == np.max(inputs["ac_voltage_peak_in"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_max", "torque_out"
        ] = np.where(inputs["torque_out"] == np.max(inputs["torque_out"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max",
            "electromagnetic_torque",
        ] = np.where(
            inputs["electromagnetic_torque"] == np.max(inputs["electromagnetic_torque"]), 1.0, 0.0
        )

        partials["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max", "rpm"] = (
            np.where(inputs["rpm"] == np.max(inputs["rpm"]), 1.0, 0.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":losses_max", "power_losses"
        ] = np.where(inputs["power_losses"] == np.max(inputs["power_losses"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_power_max",
            "shaft_power_out",
        ] = np.where(inputs["shaft_power_out"] == np.max(inputs["shaft_power_out"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_max",
            "tangential_stress",
        ] = np.where(inputs["tangential_stress"] == np.max(inputs["tangential_stress"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":surface_current_density_max",
            "surface_current_density",
        ] = np.where(
            inputs["surface_current_density"] == np.max(inputs["surface_current_density"]), 1.0, 0.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_flux_density_max",
            "air_gap_flux_density",
        ] = np.where(
            inputs["air_gap_flux_density"] == np.max(inputs["air_gap_flux_density"]), 1.0, 0.0
        )
