# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCapacitorCapacity(om.ExplicitComponent):
    """
    Computation of the capacity of the capacitor. Instead of taking the maximum of the different
    cases as in :cite:`giraud:2014`, we take the Wurst Käse scenario.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency",
            units="Hz",
            val=np.nan,
            desc="Maximum switching frequency of the IGBT modules",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ripple",
            val=1e-2,
            desc="Amplitude of the voltage ripple the capacitor should be able to withstand, "
            "as a percent of voltage caliber",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity",
            val=1e-3,
            units="F",
            desc="Capacity required to dampen the design voltage ripple",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]
        voltage_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"
        ]
        voltage_ripple_amplitude = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ripple"]
            * voltage_caliber
        )
        switching_frequency = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency"
        ]

        # As in the computation of the current caliber of the capacitor, we will take the
        # modulation index which give the greatest value of the capacity
        max_f_u = 0.3506 * current_caliber / switching_frequency

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity"
        ] = (max_f_u / voltage_ripple_amplitude)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"
        ]
        voltage_caliber = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber"
        ]
        voltage_ripple_percent = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ripple"
        ]
        switching_frequency = inputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency"
        ]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = 0.3506 / (switching_frequency * voltage_ripple_percent * voltage_caliber)
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_caliber",
        ] = (
            -0.3506
            * current_caliber
            / (switching_frequency * voltage_ripple_percent * voltage_caliber ** 2.0)
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":voltage_ripple",
        ] = (
            -0.3506
            * current_caliber
            / (switching_frequency * voltage_ripple_percent ** 2.0 * voltage_caliber)
        )
        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":switching_frequency",
        ] = (
            -0.3506
            * current_caliber
            / (switching_frequency ** 2.0 * voltage_ripple_percent * voltage_caliber)
        )
