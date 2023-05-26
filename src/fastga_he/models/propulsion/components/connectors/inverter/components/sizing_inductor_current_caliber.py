# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterInductorCurrentCaliber(om.ExplicitComponent):
    """
    Computation of the maximum current that can pass through the inductor. It is gonna be equal
    to the current caliber in one arm of the inverter but we need this component to change the
    name
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
            + ":inductor:current_caliber",
            units="A",
            val=200.0,
            desc="Current caliber of one of the inductor of the inverter",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":inductor:current_caliber",
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]
        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":inductor:current_caliber"
        ] = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber"]
