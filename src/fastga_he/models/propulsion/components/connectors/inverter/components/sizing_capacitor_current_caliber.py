# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCapacitorCurrentCaliber(om.ExplicitComponent):
    """
    Computation of the maximum current that can pass through the capacitor. Instead of taking the
    maximum of the different cases as in :cite:`giraud:2014`, we take the worst case scenario.
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

        self.add_output(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":capacitor:current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the capacitor of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        # Maximum of the term that multiplies the current caliber of the modules is found at a
        # modulation index of 0.6126 which gives a maximum of 0.46
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:current_caliber"
        ] = (
            0.46
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":capacitor:current_caliber",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = 0.46
