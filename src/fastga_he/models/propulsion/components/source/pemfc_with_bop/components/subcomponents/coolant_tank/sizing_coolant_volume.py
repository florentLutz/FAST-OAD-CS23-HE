# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingCoolantTotalVolume(om.ExplicitComponent):
    """
    The total volume of the coolant in the TMS, which is the sum of the volumes in HEX and the
    pipe total volume.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_component_names",
            default="None",
            desc="A list of the TBS components that use coolant",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_tank_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_component_names = self.options["coolant_component_names"]
        coolant_tank_id = self.options["coolant_tank_id"]

        for name in coolant_component_names:
            self.add_input(
                name="data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + name
                + ":coolant_volume",
                units="m**3",
                val=np.nan,
            )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":coolant_volume",
            units="m**3",
            val=0.001,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_tank_id = self.options["coolant_tank_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + coolant_tank_id
            + ":coolant_volume"
        ] = np.sum(inputs.values())
