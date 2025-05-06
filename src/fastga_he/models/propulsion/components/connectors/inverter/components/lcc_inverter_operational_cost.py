# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCInverterOperationalCost(om.ExplicitComponent):
    """
    Computation of the inverter annual operational cost. The lifespan expectancy is obtained from
    :cite:`cao:2023`.
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
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":cost_per_unit",
            units="USD",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost",
            units="USD/yr",
            val=350.0,
        )

        self.declare_partials(of="*", wrt="*", val=0.1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":operational_cost"] = (
            0.1
            * inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":cost_per_unit"]
        )
