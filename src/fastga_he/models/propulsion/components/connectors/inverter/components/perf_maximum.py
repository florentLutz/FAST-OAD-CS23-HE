# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum currents (DC and AC 1 phase), voltage (DC and AC peak) and
    temperature of the inverter.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        inverter_id = self.options["inverter_id"]

        self.add_input(
            "ac_current_rms_out_one_phase", units="A", val=np.full(number_of_points, np.nan)
        )
        self.add_input(
            "ac_voltage_peak_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Peak voltage on the AC side of the inverter",
        )
        self.add_input(
            "dc_current_in",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current coming in from the DC side of the inverter",
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Bus voltage on the DC side of the inverter",
        )
        self.add_input(
            "diode_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            "IGBT_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            "casing_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the heat sink",
            shape=number_of_points,
        )
        self.add_input(
            "losses_inverter",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max",
            units="A",
            val=500.0,
            desc="Maximum RMS current flowing through one arm of the inverter",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max",
            wrt="ac_current_rms_out_one_phase",
            method="exact",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max",
            units="V",
            val=800.0,
            desc="Maximum value of the peak voltage at the output of the inverter (AC side)",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max",
            wrt="ac_voltage_peak_out",
            method="exact",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_dc_max",
            units="A",
            val=500.0,
            desc="Maximum value of the current at the input of the inverter",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_dc_max",
            wrt="dc_current_in",
            method="exact",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_dc_max",
            units="V",
            val=300,
            desc="Maximum value of the voltage at the input of the inverter (DC side)",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_dc_max",
            wrt="dc_voltage_in",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:temperature_max",
            units="degK",
            val=373.15,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:temperature_max",
            wrt="IGBT_temperature",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:temperature_max",
            units="degK",
            val=373.15,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:temperature_max",
            wrt="diode_temperature",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:temperature_max",
            units="degK",
            val=373.15,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:temperature_max",
            wrt="casing_temperature",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max",
            units="W",
            val=42.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max",
            wrt="losses_inverter",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max"
        ] = np.max(inputs["ac_current_rms_out_one_phase"])
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max"
        ] = np.max(inputs["ac_voltage_peak_out"])
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_dc_max"
        ] = np.max(inputs["dc_current_in"])
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_dc_max"
        ] = np.max(inputs["dc_voltage_in"])
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:temperature_max"
        ] = np.max(inputs["IGBT_temperature"])
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:temperature_max"
        ] = np.max(inputs["diode_temperature"])
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:temperature_max"
        ] = np.max(inputs["casing_temperature"])
        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max"] = np.max(
            inputs["losses_inverter"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ac_max",
            "ac_current_rms_out_one_phase",
        ] = np.where(
            inputs["ac_current_rms_out_one_phase"]
            == np.max(inputs["ac_current_rms_out_one_phase"]),
            1.0,
            0.0,
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ac_max",
            "ac_voltage_peak_out",
        ] = np.where(
            inputs["ac_voltage_peak_out"] == np.max(inputs["ac_voltage_peak_out"]), 1.0, 0.0
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_dc_max",
            "dc_current_in",
        ] = np.where(inputs["dc_current_in"] == np.max(inputs["dc_current_in"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_dc_max",
            "dc_voltage_in",
        ] = np.where(inputs["dc_voltage_in"] == np.max(inputs["dc_voltage_in"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:temperature_max",
            "IGBT_temperature",
        ] = np.where(inputs["IGBT_temperature"] == np.max(inputs["IGBT_temperature"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:temperature_max",
            "diode_temperature",
        ] = np.where(inputs["diode_temperature"] == np.max(inputs["diode_temperature"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:temperature_max",
            "casing_temperature",
        ] = np.where(inputs["casing_temperature"] == np.max(inputs["casing_temperature"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":losses_max",
            "losses_inverter",
        ] = np.where(inputs["losses_inverter"] == np.max(inputs["losses_inverter"]), 1.0, 0.0)
