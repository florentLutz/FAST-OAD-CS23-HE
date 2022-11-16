# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCasingThermalResistance(om.ExplicitComponent):
    """
    Computation of thermal resistances between the casing and the heat sink.
    One option is based on a regression on the SEMIKRON family, the other is based on the value of
    the thermal resistance of components of the IGBT7 technology
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the inverter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
            units="V",
            val=np.nan,
            desc="Voltage caliber of the inverter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":casing:thermal_resistance",
            units="K/W",
            val=0.035,
            desc="Thermal resistance between the casing and the heat sink",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        # current_caliber = inputs[
        #     "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        # ]
        # voltage_caliber = inputs[
        #     "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"
        # ]

        # outputs[
        #     "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance"
        # ] = (0.0495 - 8.952e-6 * current_caliber - 1.555e-8 * current_caliber * voltage_caliber)
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance"
        ] = 0.021

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]
        voltage_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"
        ]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = (
            -8.952e-6 - 1.555e-8 * voltage_caliber
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
        ] = (
            -1.555e-8 * current_caliber
        )
