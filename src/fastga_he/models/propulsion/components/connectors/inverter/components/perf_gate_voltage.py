# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesGateVoltage(om.ExplicitComponent):
    """
    Reading the datasheets from Semikron, it seems like the gate voltages of the IGBT module tend
    to change significantly with temperature. A simple model based on the assumption of a linear
    variation with temperature will be taken. The default value of the coefficient linking the
    gate voltage to the temperature was computed based on the value extracted from datasheet of
    the IGBT7 technology.
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

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:gate_voltage",
            units="V",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:gate_voltage",
            units="V",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:igbt",
            val=-0.00105,
            units="degK**-1",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:diode",
            val=-0.0022,
            units="degK**-1",
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
            name="settings:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":reference_temperature",
            val=293.15,
            units="degK",
        )

        self.add_output(
            "gate_voltage_igbt",
            val=np.full(number_of_points, 1.0),
            units="V",
            shape=number_of_points,
        )
        self.add_output(
            "gate_voltage_diode",
            val=np.full(number_of_points, 1.0),
            units="V",
            shape=number_of_points,
        )

        self.declare_partials(
            of="gate_voltage_igbt",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:gate_voltage",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":properties:voltage_temperature_scale_factor:igbt",
                "IGBT_temperature",
                "settings:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":reference_temperature",
            ],
            method="exact",
        )
        self.declare_partials(
            of="gate_voltage_diode",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:gate_voltage",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":properties:voltage_temperature_scale_factor:diode",
                "diode_temperature",
                "settings:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":reference_temperature",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        diode_temperature = inputs["diode_temperature"]
        igbt_temperature = inputs["IGBT_temperature"]

        inverter_reference_temperature = inputs[
            "settings:propulsion:he_power_train:inverter:" + inverter_id + ":reference_temperature"
        ]

        alpha_v_igbt = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:igbt"
        ]
        alpha_v_diode = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:diode"
        ]

        reference_gate_voltage_igbt = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:gate_voltage"
        ]
        reference_gate_voltage_diode = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:gate_voltage"
        ]

        # To prevent reaching unfeasible value during the loops
        outputs["gate_voltage_igbt"] = np.maximum(
            reference_gate_voltage_igbt
            * (1.0 + alpha_v_igbt * (igbt_temperature - inverter_reference_temperature)),
            reference_gate_voltage_igbt / 3,
        )

        outputs["gate_voltage_diode"] = np.maximum(
            reference_gate_voltage_diode
            * (1.0 + alpha_v_diode * (diode_temperature - inverter_reference_temperature)),
            reference_gate_voltage_diode / 3,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        inverter_id = self.options["inverter_id"]

        diode_temperature = inputs["diode_temperature"]
        igbt_temperature = inputs["IGBT_temperature"]

        inverter_reference_temperature = inputs[
            "settings:propulsion:he_power_train:inverter:" + inverter_id + ":reference_temperature"
        ]

        alpha_v_igbt = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:igbt"
        ]
        alpha_v_diode = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:diode"
        ]

        reference_gate_voltage_igbt = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:gate_voltage"
        ]
        reference_gate_voltage_diode = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:gate_voltage"
        ]

        partials[
            "gate_voltage_igbt",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:gate_voltage",
        ] = 1.0 + alpha_v_igbt * (igbt_temperature - inverter_reference_temperature)
        partials[
            "gate_voltage_igbt",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:igbt",
        ] = (igbt_temperature - inverter_reference_temperature) * reference_gate_voltage_igbt
        partials[
            "gate_voltage_igbt",
            "settings:propulsion:he_power_train:inverter:" + inverter_id + ":reference_temperature",
        ] = (
            -reference_gate_voltage_igbt * alpha_v_igbt
        )
        partials["gate_voltage_igbt", "IGBT_temperature"] = np.eye(number_of_points) * (
            reference_gate_voltage_igbt * alpha_v_igbt
        )

        partials[
            "gate_voltage_diode",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:gate_voltage",
        ] = 1.0 + alpha_v_diode * (diode_temperature - inverter_reference_temperature)
        partials[
            "gate_voltage_diode",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":properties:voltage_temperature_scale_factor:diode",
        ] = (diode_temperature - inverter_reference_temperature) * reference_gate_voltage_diode
        partials[
            "gate_voltage_diode",
            "settings:propulsion:he_power_train:inverter:" + inverter_id + ":reference_temperature",
        ] = (
            -alpha_v_diode * reference_gate_voltage_diode
        )
        partials["gate_voltage_diode", "diode_temperature"] = np.eye(number_of_points) * (
            alpha_v_diode * reference_gate_voltage_diode
        )
