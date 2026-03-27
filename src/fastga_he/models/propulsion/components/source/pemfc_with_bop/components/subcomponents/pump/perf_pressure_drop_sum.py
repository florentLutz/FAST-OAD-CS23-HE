# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCoolantSystemPressureDrop(om.ExplicitComponent):
    """
    The sum of the coolant pressure drops of the components.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pump_id",
            default=None,
            desc="Identifier of the pump",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_component_ids",
            default="None",
            desc="A list of the TBS components that use coolant",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_component_ids = self.options["coolant_component_ids"]
        pump_id = self.options["pump_id"]

        for coolant_component_id in coolant_component_ids:
            self.add_input(
                name="data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + coolant_component_id
                + ":coolant_pressure_drop",
                units="Pa",
                val=np.nan,
            )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pressure_compensation",
            units="Pa",
            val=1e5,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pump_id = self.options["pump_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pressure_compensation"
        ] = np.sum(inputs.values()) + 1e5  # This is to consider the pressure drop from the valve
