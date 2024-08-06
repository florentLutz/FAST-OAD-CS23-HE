# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRectifierCapacitorCapacity(om.ExplicitComponent):
    """
    Computation of the capacity of the capacitor. Instead of taking the maximum of the different
    cases as in :cite:`giraud:2014`, we take the worst case scenario.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the rectifier",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
            units="V",
            val=np.nan,
            desc="Voltage caliber of the rectifier",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":switching_frequency",
            units="Hz",
            val=np.nan,
            desc="Maximum switching frequency of the IGBT modules",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ripple",
            val=1e-2,
            desc="Amplitude of the voltage ripple the capacitor should be able to withstand, "
            "as a percent of voltage caliber",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity",
            val=1e-3,
            units="F",
            desc="Capacity required to dampen the design voltage ripple",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        current_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]
        voltage_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
        ]
        voltage_ripple_amplitude = (
            inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ripple"]
            * voltage_ac_caliber
        )
        switching_frequency = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency"
        ]

        # As in the computation of the current caliber of the capacitor, we will take the
        # modulation index which give the greatest value of the capacity
        max_f_u = 0.3506 * current_ac_caliber / switching_frequency

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity"
        ] = max_f_u / voltage_ripple_amplitude

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        current_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber"
        ]
        voltage_ac_caliber = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber"
        ]
        voltage_ripple_percent = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ripple"
        ]
        switching_frequency = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency"
        ]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":current_ac_caliber",
        ] = 0.3506 / (switching_frequency * voltage_ripple_percent * voltage_ac_caliber)
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ac_caliber",
        ] = (
            -0.3506
            * current_ac_caliber
            / (switching_frequency * voltage_ripple_percent * voltage_ac_caliber**2.0)
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":voltage_ripple",
        ] = (
            -0.3506
            * current_ac_caliber
            / (switching_frequency * voltage_ripple_percent**2.0 * voltage_ac_caliber)
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":capacitor:capacity",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":switching_frequency",
        ] = (
            -0.3506
            * current_ac_caliber
            / (switching_frequency**2.0 * voltage_ripple_percent * voltage_ac_caliber)
        )
