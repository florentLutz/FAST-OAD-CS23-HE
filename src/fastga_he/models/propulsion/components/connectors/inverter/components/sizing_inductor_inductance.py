# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterInductorInductance(om.ExplicitComponent):
    """
    Computation of the inductance of the inductor, implementation of the formula from
    :cite:`giraud:2014`.
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
        self.add_input(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":inductor:short_circuit_time",
            units="s",
            val=2.0e-6,
            desc="Duration of the short circuit",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ripple",
            val=0.2,
            desc="Amplitude of the current ripple as a percent of the current caliber",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance",
            units="H",
            val=0.22e-6,
            desc="Inductance of the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]
        current_ripple = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ripple"
        ]
        voltage_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"
        ]
        delta_t_des = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":inductor:short_circuit_time"
        ]

        inductance = voltage_caliber * delta_t_des / (current_caliber * current_ripple)

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance"
        ] = inductance

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]
        current_ripple = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ripple"
        ]
        voltage_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"
        ]
        delta_t_des = inputs[
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":inductor:short_circuit_time"
        ]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = -voltage_caliber * delta_t_des / (current_caliber**2.0 * current_ripple)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_ripple",
        ] = -voltage_caliber * delta_t_des / (current_caliber * current_ripple**2.0)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
        ] = delta_t_des / (current_caliber * current_ripple)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:inductance",
            "data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":inductor:short_circuit_time",
        ] = voltage_caliber / (current_caliber * current_ripple)
